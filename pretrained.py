from torch import nn
from torchvision import models


# Vzít VGG předtrénovanou síť 
class PretrainedResNet50(nn.Module):
    def __init__(self, num_classes, emb_size):
        super(PretrainedResNet50, self).__init__()
        self.pretrainedModel = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*(list(self.pretrainedModel.children())[:-1]))
        for param in self.features.parameters(): 
            param.requires_grad = False

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(2048, emb_size)
        self.fc2 = nn.Linear(emb_size, num_classes)
        # Linear(in_features=2048, out_features=512)

    def encode(self, x):
        x = self.features(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

    def forward(self, x):
        x = self.encode(x)   
        x = self.relu(x) 
        x = self.fc2(x)
        return x

    

