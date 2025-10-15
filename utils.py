# utils.py

import os
import random
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

def seed_everything(seed):
    """
    Seeds all relevant random number generators for reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    
    # These two settings are crucial for deterministic results with CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # A more recent PyTorch version feature to enforce determinism
    # This will raise an error if a non-deterministic function is used
    try:
        torch.use_deterministic_algorithms(True)
        # For certain operations, this environment variable is also needed
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    except Exception as e:
        print(f"Could not enforce deterministic algorithms: {e}")

def seed_worker(worker_id):
    """
    Seeds each DataLoader worker individually.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def denormalize(tensor, config):
    """Reverses the normalization on a tensor image for visualization."""
    tensor = tensor.clone()
    mean = config['dataloader_params']['mean']
    std = config['dataloader_params']['std']
    mean = torch.tensor(mean, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(3, 1, 1)
    tensor.mul_(std).add_(mean)
    tensor = torch.clamp(tensor, 0, 1)
    return tensor

def binary_mask_to_vis_image(mask, lesion_color=[0, 255, 0]):
    """Converts a binary segmentation mask to a colored PIL Image."""
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    if mask.ndim == 3:
        mask = np.squeeze(mask, axis=0)
    
    segment_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    segment_image[mask == 1] = lesion_color
    return Image.fromarray(segment_image)

def plot_result(config, image, prediction_logits, ground_truth_mask):
    """Plots original image, ground truth, and predicted segmentation."""
    denormalized_image = denormalize(image,config)
    preds_binary = (torch.sigmoid(prediction_logits) > 0.5).float()
    
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(denormalized_image.permute(1, 2, 0).cpu())
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(binary_mask_to_vis_image(ground_truth_mask))
    plt.title("Ground Truth Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(binary_mask_to_vis_image(preds_binary))
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.show()

def load_config(config_path="config.yaml"):
    """Loads the YAML config file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config