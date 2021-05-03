from torch import nn
import torch
import torch.nn.functional as F

#outputs [B,10] targets [B,]
class ArcFace(nn.Module):
    def __init__(self, margin = 0.5, scale = 64):
        super(ArcFace, self).__init__()
        self.margin = margin
        self.scale = scale

    def forward(self, outputs, targets): #0.5 64
        criterion = nn.CrossEntropyLoss()
        original_target_logits = outputs.gather(1, torch.unsqueeze(targets, 1))
        # arccos grad divergence error https://github.com/pytorch/pytorch/issues/8069
        #eps = 1e-7 
        #original_target_logits = torch.clamp(original_target_logits, -1+eps, 1-eps)
        original_target_logits = original_target_logits * 0.999999
        thetas = torch.acos(original_target_logits) 
        marginal_target_logits = torch.cos(thetas + self.margin)

        #f=original_target_logits-marginal_target_logits
        #print(min(f),max(f))
        #print('orig',original_target_logits)
        #print('mafg',marginal_target_logits)

        one_hot_mask = F.one_hot(targets, num_classes=outputs.shape[1])
        
        diff = marginal_target_logits - original_target_logits
        #diff = torch.clip(diff,-1,0)
        expanded = diff.expand(-1, outputs.shape[1])
        outputs = self.scale * (outputs + (expanded * one_hot_mask))
        return criterion(outputs, targets)