from torch.utils.data import Dataset
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