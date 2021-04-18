
import torch
import os
from custom_dataset import CustomDataset
from torchvision.transforms import ToTensor, ToPILImage, Lambda, Compose, Resize
from log import Log
from torch.utils.data import DataLoader, Dataset
import calendar
import time

def get_datasets(batch_size, batch_size_test = 128):
    transform=Compose([Resize(96), ToTensor()]) 
    #CASIA DATASET
    dataset = CustomDataset(csv_file = 'datasets/CASIA/casia_train.csv', root_dir = 'datasets/CASIA', transform = transform) 
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataset = CustomDataset(csv_file = 'datasets/CASIA/casia_test.csv', root_dir = 'datasets/CASIA', transform = transform) 
    test_dataloader = DataLoader(dataset, batch_size=batch_size_test, shuffle=True)
    dataset = CustomDataset(csv_file = 'datasets/CASIA/casia_pairs.csv', root_dir = 'datasets/CASIA', transform = transform, pairs=True) 
    casia_pairs_dataloader = DataLoader(dataset)
    #LFW DATASET
    dataset = CustomDataset(csv_file = 'datasets/LFW/lfw_pairs.csv', root_dir = 'datasets/LFW', transform = transform, pairs=True) 
    lfw_pairs_dataloader = DataLoader(dataset)
    return (train_dataloader, test_dataloader, casia_pairs_dataloader, lfw_pairs_dataloader)

def create_logs():
    path = "outputs"
    logs_names = ["acc","test_acc","ver","lfw_ver","loss","tf","lfw_tf"]
    logs = []
    for ln in logs_names:
        log = Log(os.path.join(path, ln))
        logs.append(log)
    return logs
