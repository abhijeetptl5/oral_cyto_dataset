import torch.nn as nn
import torch

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        # Apply sigmoid to prediction
        pred = torch.sigmoid(pred)
        
        # Flatten predictions and targets
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        # Calculate Dice coefficient and loss
        dice_coeff = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_coeff
        
        return dice_loss

# Combined loss
class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_criterion = nn.BCEWithLogitsLoss()
        self.dice_criterion = DiceLoss()
        
    def forward(self, pred, target):
        bce_loss = self.bce_criterion(pred, target)
        dice_loss = self.dice_criterion(pred, target)
        combined_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        
        return combined_loss, bce_loss, dice_loss