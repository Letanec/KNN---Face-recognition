from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, ToPILImage, Lambda, Compose, Resize
import pandas as pd
import os
import torch
from PIL import Image

class CustomDataset(Dataset):
  def __init__(self, csv_file, root_dir, transform=None, pairs=False):
    self.annotations = pd.read_csv(csv_file, header=None)
    self.root_dir = root_dir
    self.transform = transform
    if pairs:
      self.format = ['image', 'image', 'label']
    else:
      self.format = ['image', 'label']

  def __len__(self):
    return len(self.annotations)
  
  def __getitem__(self, index):
    result = []
    for column, item in enumerate(self.format):
      if item == 'image':
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, column])
        image = Image.open(img_path)
        if self.transform:
          image = image.convert("RGB")
          image = self.transform(image)
        result.append(image)
      elif item == 'label':
        label = torch.tensor(int(self.annotations.iloc[index,column]))
        result.append(label)
      else:
        raise Exception('Unknown format.') 
    return result

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