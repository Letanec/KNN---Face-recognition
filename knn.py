import os
import torch
import torch.nn.functional as f
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, ToPILImage, Lambda, Compose, Resize
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.manifold import TSNE
import pandas as pd
from PIL import Image
from scipy.spatial import distance
from sklearn.datasets import fetch_lfw_pairs
import calendar;
import time;

from res_net import ResNet18, ResNet50
from casia_dataset import CasiaDataset
from validation import validate, print_ROC

batch_size = 8
model_path = "./model.pt"
num_classes = 10575   
embeding_size = 512 #Resnet18: 512 # resnet 50: 2048

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

#LFW DATASET
fetch_lfw_pairs = fetch_lfw_pairs(subset='10_folds', funneled=False, color=True, resize=224/250, slice_=None)
lfw_pairs = fetch_lfw_pairs.pairs
lfw_labels = fetch_lfw_pairs.target
lfw_pairs = lfw_pairs.transpose((0, 1, 4, 2, 3))
lfw_pairs = Tensor(lfw_pairs)
print(lfw_pairs.shape)

#CASIA DATASET
transform=Compose([Resize(224), ToTensor()]) 
dataset = CasiaDataset(csv_file = '../../CASIA/casia2.csv', root_dir = '../../CASIA', transform = transform) 
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
short_len = 10000
dataset_short, _ = torch.utils.data.random_split(dataset, [short_len, len(dataset) - short_len])
train_dataloader_short = DataLoader(dataset_short, batch_size=batch_size, shuffle=True)

model = ResNet18(num_classes=num_classes, emb_size=embeding_size).to(device)
#model = ResNet50(num_classes=num_classes, emb_size=embeding_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1) 

train_print_period = 10
test_period = 100

accs = losses = test_vers =  []

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Create output files
ts = calendar.timegm(time.gmtime())
TARS_and_FARs_file = open(f"./outputs/all_TARs_and_FARs_{ts}.out", "w").close()
test_vers_file = open(f"./outputs/test_accs{ts}.out", "w").close()
train_accs_file = open(f"./outputs/train_accs{ts}.out", "w").close()
loss_file = open(f"./outputs/loss{ts}.out", "w").close()

#training
i = 0
best_test_ver = 0
for e in range(100): 
    model.train()
    loss_sum = correct = total = 0
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
        
        if i>0 and i % train_print_period == 0:
            print("iteration:", i, "epoch:", e, "batch:", b, "avg loss:", "%.3f" % avg_loss, "epoch avg train acc:", train_acc)
            with open(f"./outputs/train_accs{ts}.out", "a") as train_accs_file:
                train_accs_file.write(f'{train_acc}\n')
            with open(f"./outputs/loss{ts}.out", "a") as loss_file:
                loss_file.write(f'{avg_loss}\n')
            accs.append(train_acc)
            losses.append(avg_loss)

        if i>0 and i%test_period == 0:
            test_ver = validate(model, lfw_pairs, lfw_labels, device, ts)
            model.train()
            print("epoch:", e, "test ver: ", test_ver)
            test_vers.append(test_ver)

            if test_ver > best_test_ver:  
                best_test_ver = test_ver 
                print_ROC(model, lfw_pairs, lfw_labels, device)         
                torch.save(model.state_dict(), model_path)

        if i == 20000 or i == 28000:
            optimizer.param_groups[0]['lr'] /= 10

        if i == 32000:
            break

        i+=1

TARS_and_FARs_file.close()
test_vers_file.close()



print("acc")
plt.plot(accs)
plt.show()
print("loss")
plt.plot(losses)
plt.show()
print("test vers")
plt.plot(test_vers)
plt.show()
