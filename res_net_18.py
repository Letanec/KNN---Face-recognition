import os
import torch
import torch.nn.functional as f
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose, Resize
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.manifold import TSNE
import pandas as pd
from PIL import Image
from google.colab import drive

batch_size = 8
epoch_num = 1000
model_path = "model.pt"
num_classes = 1700 #casia 10575  
lr = 0.001
wd = 0.0001
target_test_acc = 80
visualization_cnt = 10
visualization_colors = ["b", "g", "r", "c", "m", "y", "k", "w", "orange", "gray"]

#drive.mount('/content/drive/')
#ODKOMENTOVAT JEDNO PRI PRVOM SPUSTENI
#CASIA dataset ako zip, musi byt uploadnuty vo vasom google drive v zlozke KNN
#!unzip /content/drive/MyDrive/KNN/CASIA.zip
#LFW dataset ako zip, musi byt uploadnuty vo vasom google drive v zlozke KNN
#!unzip /content/drive/MyDrive/KNN/lfw-deepfunneled.zip

class LFWDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
  
    def __len__(self):
        return len(self.annotations)
  
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index,1]))

        if self.transform:
            image = image.convert("RGB") 
            image = self.transform(image)
    
        return (image, y_label)

transform=Compose([Resize(224), ToTensor()]) 

# ODKOMENTOVAT JEDNO
# LFW dataset
dataset = LFWDataset(csv_file = 'lfw-deepfunneled/lfw-deepfunneled/lfw.csv', root_dir = 'lfw-deepfunneled/lfw-deepfunneled', transform = transform) 
# CASIA dataset
#dataset = LFWDataset(csv_file = 'CASIA/casia2.csv', root_dir = 'CASIA', transform = transform)

train_size = int(len(dataset)*0.8)
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_set, batch_size=1000, shuffle=True)

# Vizualization data
first = True
for X, Y in train_dataloader:
  mask = np.vectorize(lambda x: x < visualization_cnt)(Y)
  if first:
    visualization_data_X,visualization_data_Y = X[mask],Y[mask]
    first = False
  else:
    visualization_data_X = np.vstack((visualization_data_X,X[mask]))
    visualization_data_Y = np.hstack((visualization_data_Y,Y[mask]))
visualization_data = torch.from_numpy(visualization_data_X), torch.from_numpy(visualization_data_Y)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

#Input stem block
class InputStemBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int):
        super(InputStemBlock, self).__init__()
        self.operation = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=7, stride=2, padding=3, bias=True),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
    def forward(self, x):
        return self.operation(x)

#Residual block
class ResidualBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, stride: int):
        super(ResidualBlock, self).__init__()
        self.operation = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(),
            #nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(channels_out, channels_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(channels_out)
        )
        #results of self.operation and self.identity have to have same C,W,H to perform addition
        if channels_in == channels_out and stride == 1:
            #no change needed
            self.identity = nn.Sequential()
        else:
            #correct C,W,H by 1x1 conv
            self.identity = nn.Sequential(
                nn.Conv2d(channels_in, channels_out, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(channels_out)
            )        
    def forward(self, x):
        output = self.operation(x) + self.identity(x)
        output = f.relu(output)
        return output

#ResNet18
class ResNet18(nn.Module):
    def __init__(self, num_classes: int):
        super(ResNet18, self).__init__()
        self.operation = nn.Sequential(
            #input stem
            InputStemBlock(3,64),     #3x224x224 -> 64x56x56
            #1 residual layer 
            ResidualBlock(64,64,1),   #64x56x56 -> 64x56x56
            ResidualBlock(64,64,1),   #64x56x56 -> 64x56x56
            #2 residual layer
            ResidualBlock(64,128,2),  #64x56x56  -> 128x28x28
            ResidualBlock(128,128,1), #128x28x28 -> 128x28x28
            #3 residual layer
            ResidualBlock(128,256,2), #128x28x28 -> 256x14x14
            ResidualBlock(256,256,1), #256x14x14 -> 256x14x14
            #4 residual layer
            ResidualBlock(256,512,2), #256x14x14 -> 512x7x7
            ResidualBlock(512,512,1), #512x7x7   -> 512x7x7
            #fully connected
            nn.AvgPool2d(7),  #512x7x7 -> 512x1x1
            nn.Flatten(),
            nn.Linear(512, 1000),
            nn.ReLU()
        )
        self.softmax =  nn.Sequential(
            nn.Linear(1000, num_classes),
            nn.Softmax(dim=1)
        )

    def encode(self, x):
        return self.operation(x)

    def forward(self,x):
        x = self.operation(x)
        x = self.softmax(x)
        return x

def validate(model):
    model.eval()
    with torch.no_grad():
        correct = total = 0 
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = torch.argmax(outputs.data, dim=1)
            correct += predicted.eq(labels.data).cpu().sum()
            total += labels.size(0)
    return (100. * correct / total).item()

def visualize_embeding(model):  
    model.eval()
    with torch.no_grad():
        inputs,labels = visualization_data
        inputs,labels = inputs.to(device), labels.to(device)
        outputs = model.encode(inputs)
        outputs_embedded = TSNE(n_components=2).fit_transform(outputs.cpu().detach().numpy())
        x,y = np.hsplit(outputs_embedded, 2)
        x,y = x.flatten(), y.flatten()
        for i in range(len(x)):
            plt.scatter(x[i], y[i], color=visualization_colors[labels[i]])
        plt.show()


model = ResNet18(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd) 

#training
#foreach epoch
for e in range(epoch_num):
    model.train()
    loss_sum = correct = total = 0
    #foreach batch
    for b, (inputs, labels) in enumerate(train_dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        avg_loss = loss_sum / (b+1)

        predicted = torch.argmax(outputs.data, dim=1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum() 
        train_acc = (100. * correct / total).item()
        
        if b%3 == 0:
          print("epoch:", e, "batch:", b, "avg loss:", "%.3f" % avg_loss, "epoch avg train acc:", train_acc)

    test_acc = validate(model)
    print("epoch:", e, "test acc: ", test_acc)
    if test_acc > target_test_acc:
        break

torch.save(model.state_dict(), model_path)
visualize_embeding(model)

