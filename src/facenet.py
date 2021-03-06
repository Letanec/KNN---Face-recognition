import torch
from torch import nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
from datetime import datetime
class Pretrained_Facenet(nn.Module):
    def __init__(self, arcface=False):
        super(Pretrained_Facenet, self).__init__()    
        self.arcface = arcface
        self.pretrained_model = InceptionResnetV1(classify=True, pretrained='casia-webface') 
        if arcface:
            self.last_fc = list(self.pretrained_model.children())[-1]
            self.last_fc.bias = None

    def save(self, path, train_acc=0):
        ts = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
        model_path = path + "/model_" + ts +"_acc_" + '{:.3f}'.format(train_acc)
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