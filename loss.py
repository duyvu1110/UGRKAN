import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input_sig = torch.sigmoid(input)
        num = target.size(0)
        input_flat = input_sig.view(num, -1)
        target_flat = target.view(num, -1)
        intersection = (input_flat * target_flat)
        dice = (2. * intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        dice_loss = 1 - dice.sum() / num
        return 0.5 * bce + dice_loss