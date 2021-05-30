import torch
import torch.nn.functional as F
from torch import nn


class ArcFace(nn.Module):
    def __init__(self, margin = 0.5, scale = 64):
        super(ArcFace, self).__init__()
        self.margin = margin
        self.scale = scale

    #implementovano dle popisu z clanku ArcaFace https://arxiv.org/pdf/1801.07698.pdf
    def forward(self, outputs, targets): #0.5 64
        criterion = nn.CrossEntropyLoss()
        original_target_logits = outputs.gather(1, torch.unsqueeze(targets, 1))
        #arccos divergence pro hodnoty kolem hranice def. oboru -> *0.999999
        original_target_logits = original_target_logits * 0.999999
        thetas = torch.acos(original_target_logits) 
        marginal_target_logits = torch.cos(thetas + self.margin)
        one_hot_mask = F.one_hot(targets, num_classes=outputs.shape[1])
        diff = marginal_target_logits - original_target_logits
        expanded = diff.expand(-1, outputs.shape[1])
        outputs = self.scale * (outputs + (expanded * one_hot_mask))
        return criterion(outputs, targets)
