# metrics.py

import torch
from medpy.metric.binary import jc, dc, hd95

def iou_score(output, target):
    """Calculates Intersection over Union (IoU) and other metrics."""
    smooth = 1e-5
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    
    output_ = output > 0.5
    target_ = target > 0.5
    
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    
    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * iou) / (iou + 1)

    try:
        hd95_val = hd95(output_, target_)
    except:
        hd95_val = 0
    
    return iou, dice, hd95_val

def dice_coefficient(pred, target, eps=1e-6):
    """Computes Dice coefficient for binary masks."""
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    
    dice = (2. * intersection + eps) / (union + eps)
    return dice.mean().item()