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
# drive.mount('/content/drive/')
# !unzip /content/drive/MyDrive/KNN/lfw-deepfunneled.zip

transform=Compose([
    Resize(224), #28x28 -> 224x224
    ToTensor(),
    Lambda(lambda x: x.repeat(3, 1, 1))  #3x copy grayscale channel -> "RGB"  
]) 

# Download training data from open datasets
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

# Download test data from open datasets
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

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
      image = self.transform(image)
    
    return (image, y_label)

dataset = LFWDataset(csv_file = 'lfw-deepfunneled/lfw.csv', root_dir = 'lfw-deepfunneled', transform = transform)
train_set = dataset

batch_size = 64
epoch_num = 1000
lr = 0.1
visualization_cnt = 5
visualization_colors = ["green", "blue", "red", "black", "yellow"]

train_set_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
#test_set_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

# Create data loaders
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=100)

# Vizualization data
X, Y = next(iter(test_dataloader))
mask = np.vectorize(lambda x: x < visualization_cnt)(Y)
visualization_data = X[mask],Y[mask]

'''
for X, y in train_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break
'''

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

#Input stem block
class InputStemBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int):
        super(InputStemBlock, self).__init__()
        self.operation = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=7, stride=2, padding=3, bias=False),
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
            nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(),
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels_out)
        )
        #results of self.operation and self.identity have to have same C,W,H to perform addition
        if channels_in == channels_out and stride == 1:
            #no change needed
            self.identity = nn.Sequential()
        else:
            #correct C,W,H by 1x1 conv
            self.identity = nn.Sequential(
                nn.Conv2d(channels_in, channels_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels_out)
            )        
    def forward(self, x):
        output = self.operation(x) + self.identity(x)
        output = f.relu(output)
        return output

#Conv block, resnet50
class ConvBlock(nn.Module):
    def __init__(self, batch_size: int, channels_in: int, channels_out: int):
        super(NeuralNetwork, self).__init__()
        self.conv1x1a = nn.Conv2d(self.channels_in, self.channels_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1x1b = nn.Conv2d(self.channels_in, self.channels_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1x1c = nn.Conv2d(self.channels_in, self.channels_out, kernel_size=1, stride=1, padding=0, bias=False)
        #Do batch_norm patří batch_size, nebo channels_out?
        self.batch_norm = nn.BatchNorm2d(batch_size)
        self.conv3x3 = nn.Conv2d(self.channels_in, self.channels_out, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu = nn.ReLU()

    def forward (self, input: Tensor) -> Tensor:
        output = self.conv1x1a(input)
        output = self.batch_norm(output)
        output = self.relu(output)

        forwarded_output = self.conv1x1b(input)
        forwarded_output = self.batch_norm(forwarded_output)

        output = self.conv3x3(output)
        output = self.batch_norm(output)
        output = self.relu(output)

        output = self.conv1x1c(output)
        output = self.batch_norm(output)

        output = torch.add(output, forwarded_output)
        output = self.relu(output)

        return (output)

#Identity block, resnet50
class IdentityBlock(nn.Module):
    def __init__(self, batch_size: int, channels_in: int, channels_out: int):
        super(NeuralNetwork, self).__init__()
        self.conv1x1a = nn.Conv2d(self.channels_in, self.channels_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1x1b = nn.Conv2d(self.channels_in, self.channels_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1x1c = nn.Conv2d(self.channels_in, self.channels_out, kernel_size=1, stride=1, padding=0, bias=False)
        #Do batch_norm patří batch_size, nebo channels_out?
        self.batch_norm = nn.BatchNorm2d(batch_size)
        self.conv3x3 = nn.Conv2d(self.channels_in, self.channels_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()

    def forward (self, input: Tensor) -> Tensor:
        output = self.conv1x1a(input)
        output = self.batch_norm(output)
        output = self.relu(output)

        output = self.conv3x3(output)
        output = self.batch_norm(output)
        output = self.relu(output)

        output = self.conv1x1c(output)
        output = self.batch_norm(output)
        
        output = torch.add(output, input)
        output = self.relu(output)

        return (output)

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
    model.train(mode=False)
    correct = total = 0 
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        predicted = torch.argmax(outputs.data, dim=1)
        correct += predicted.eq(labels.data).cpu().sum()
        total += labels.size(0)
    return (100. * correct / total).item()

def visualize_embeding(model):  
    inputs,labels = visualization_data
    model.train(mode=False)
    outputs = model.encode(inputs)
    outputs_embedded = TSNE(n_components=2).fit_transform(outputs.detach().numpy())
    x,y = np.hsplit(outputs_embedded, 2)
    x,y = x.flatten(), y.flatten()
    for i in range(len(x)):
        plt.scatter(x[i], y[i], color=visualization_colors[labels[i]])
    plt.show()

model = ResNet18(10).to(device)
criterion = nn.CrossEntropyLoss ()
optimizer = optim.SGD(model.parameters(), lr=lr) 

visualize_embeding(model)
exit()

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
        predicted = torch.argmax(outputs.data, dim=1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()
        avg_loss = loss_sum / (b+1)
        train_acc = (100. * correct / total).item()
        test_acc = validate(model)
        print("epoch:", e, "batch:", b, "avg loss:", "%.3f" % avg_loss, "train acc:", train_acc, "test acc: ", test_acc)

