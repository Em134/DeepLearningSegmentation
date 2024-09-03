import torch.nn as nn
import torch
import torch.nn.functional as F


class BinaryMaskLoss(nn.Module):
    def __init__(self, weight=0.8, size_average=True):
        super(BinaryMaskLoss, self).__init__()
        self.weight = weight

    def forward(self, inputs, targets, smooth=1):
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        dice_score = 1-dice

        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = 0.8 * (1 - BCE_EXP)**2 * BCE
                       
        return self.weight * dice_score + (1 - self.weight) * focal_loss