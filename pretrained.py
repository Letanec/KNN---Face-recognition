from torch import nn
import torch
from torchvision import models
import torch.nn.functional as F
import time
import calendar

PRETRAINED_FEATURES_OUT = 2048

# Vzít VGG předtrénovanou síť 
class PretrainedResNet50(nn.Module):
    def __init__(self, num_classes, emb_size):
        super(PretrainedResNet50, self).__init__()
        self.pretrainedModel = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*(list(self.pretrainedModel.children())[:-1]))
        # for param in self.features.parameters(): 
        #     param.requires_grad = False
        
        #self.dropout = nn.Dropout(p=0.5, inplace=False)
        self.batchNorm1d = nn.BatchNorm1d(emb_size)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(PRETRAINED_FEATURES_OUT, emb_size)
        self.fc2 = nn.Linear(emb_size, num_classes, bias=False)

    def save_model(self, train_acc):
        
        model_path = "./models/resnet50_acc_" + '{:.3f}'.format(train_acc) + "_time_" + str(calendar.timegm(time.gmtime()))
        torch.save(self.state_dict(), model_path)
        
    def encode(self, x):
        x = self.features(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.batchNorm1d(x)    
        
        return x

    def forward(self, x):
        x = self.encode(x) 
        x_norm = F.normalize(x, p=2, dim=1) * 0.877
        w_norm = F.normalize(self.fc2.weight, p=2, dim=1) * 0.877
        logits = x_norm.matmul(w_norm.T)
        #x = self.relu(logits) 
        return logits

    

