
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
import numpy as np
from log import Log
from evaluation import test
from arc_face import ArcFace
from datasets import prepare_datasets

def train(
    model, 
    device, 
    arcface, 
    train_loader, 
    test_loader, 
    log_dir, 
    model_dir,
    train_acc_interval = 100, 
    test_acc_interval = 10000):
    
    iter = 0
    train_acc = 0
    log = Log(log_dir)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  
    if arcface:
        criterion = ArcFace()
        print("Using loss function ArcFace")
    else: 
        criterion = nn.CrossEntropyLoss()
        print("Using loss function CrossEntropy")

    for epoch in range(epochs_num): 
        loss_sum = correct = total = 0
        for batch, (inputs, labels) in enumerate(train_loader):    
            inputs, labels = inputs.to(device), labels.to(device)
            model.train()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            avg_loss = loss_sum / (batch+1)   

            predicted = torch.argmax(outputs.data, dim=1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            train_acc = (correct / total).item()

            if iter%train_acc_interval == 0:
                print('iter',iter,'epoch',epoch,'train acc',train_acc,'loss',avg_loss)
                log.train(train_acc)
                log.loss(avg_loss)

            if iter != 0 and iter%test_acc_interval == 0:
                test_acc = test(model, test_loader, device)
                print('iter',iter,'epoch',epoch,'test acc',test_acc)
                log.test(test_acc)

            iter += 1

        model.save(model_dir, train_acc)