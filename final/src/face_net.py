from datetime import datetime

import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
from torch import nn


class Pretrained(nn.Module):
    def __init__(self, arcface=False):
        super(Pretrained, self).__init__()    
        self.arcface = arcface
        self.pretrained_model = InceptionResnetV1(classify=True, pretrained='casia-webface') 
        if arcface:
            self.last_fc = list(self.pretrained_model.children())[-1]
            self.last_fc.bias = None

    def save(self, path, train_acc=0):
        ts = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
        model_path = path + "/model_" + ts +"_acc_" + '{:.3f}'.format(train_acc) + ".pt"
        torch.save(self.state_dict(), model_path)
        
    def encode(self, x):
        self.pretrained_model.classify = False
        x = self.pretrained_model(x)
        return x

    def forward(self, x):
        if self.arcface: 
            x = self.encode(x) 
            weight = F.normalize(self.last_fc.weight, p=2, dim=1) 
            logits = x.matmul(weight.T)
        else:
            self.pretrained_model.classify = True
            logits = self.pretrained_model(x)
        return logits
