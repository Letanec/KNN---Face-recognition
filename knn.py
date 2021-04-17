import os
import torch
import torch.nn.functional as f
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, ToPILImage, Lambda, Compose, Resize
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from PIL import Image
from scipy.spatial import distance
import calendar
import time
from res_net import ResNet18, ResNet50
from validation import verify, validate, tars_fars, print_ROC
from helpers import get_datasets, create_logs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", help="set batch size", action="store", type=int, default=32)
    parser.add_argument("-t", "--train_print_period", help="set period of printing training progress", action="store", type=int, default=10)
    parser.add_argument("-v", "--test_period", help="set period of validation", action="store", type=int, default=100)
    parser.add_argument("-v", "--ver_period", help="set period of verification", action="store", type=int, default=1500)
    parser.add_argument("-v", "--lfw_ver_period", help="set period of lfw verification", action="store", type=int, default=6000)  
    args = parser.parse_args()
    
    # Constants
    batch_size = args.batch_size
    train_print_period = args.train_print
    test_period = args.validation_period
    ver_period = args.ver_period
    lfw_ver_period = args.lfw_ver_period
    model_path = "./model.pt" 
    num_classes = 10575   
    embeding_size = 512    
    accs = losses = test_vers =  []

    # Get cpu or gpu device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    # Datasets
    train_dataloader, test_dataloader, casia_pairs_dataloader, lfw_pairs_dataloader = get_datasets(batch_size)
    
    # Logs
    acc_log, test_acc_log, ver_log, lfw_ver_log, loss_log, tf_log = create_logs()

    # Model, criterionm, optimizer
    model = ResNet50(num_classes=num_classes, emb_size=embeding_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) 

    # Training
    i = 0
    best_ver = 0
    for e in range(6): 
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
            
            if i>0 and i % train_print_period == 0:
                train_acc = (correct / total).item()
                acc_log.print(i,e,train_acc)
                loss_log.print(i,e,avg_loss)

            if i>0 and i % test_period == 0:
                test_acc = validate(model, test_dataloader, device)
                model.train()
                test_acc_log.print(i,e,test_acc)

            if i>0 and i % ver_period == 0:
                ver = verify(model, casia_pairs_dataloader, device)
                tf = tars_fars(model, casia_pairs_dataloader, device)
                model.train() 
                ver_log.print(i,e,ver)
                tf_log.print(i,e,tf)
                print_ROC(tf[0], tf[1])
                if ver > best_ver:  
                    best_ver = ver     
                    torch.save(model.state_dict(), model_path)
                

            if i>0 and i % lfw_ver_period == 0:
                lfw_ver = verify(model, lfw_pairs_dataloader, device)
                lfw_tf = tars_fars(model, lfw_pairs_dataloader, device)
                model.train()
                lfw_ver_log.print(i,e,lfw_ver)
                lfw_tf_log.print(i,e,lfw_tf)
                print_ROC(lfw_tf[0], lfw_tf[1])

            i+=1


if __name__ == "__main__":
    main()