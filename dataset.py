# dataset.py

import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import seed_worker, calculate_mean_std

class BUSIDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths, self.mask_paths, self.labels = [], [], []
        self.class_names = ["benign", "malignant"]
        for class_name in self.class_names:
            class_dir = os.path.join(self.root_dir, class_name)
            for img_path in glob.glob(os.path.join(class_dir, "*.png")):
                if "_mask" not in img_path:
                    mask_path = img_path.replace(".png", "_mask.png")
                    if os.path.exists(mask_path):
                        self.image_paths.append(img_path)
                        self.mask_paths.append(mask_path)
                        self.labels.append(class_name)
        assert len(self.image_paths) != 0, "Images path is empty"
        assert len(self.image_paths) == len(self.mask_paths), "Mismatch in image and mask counts."

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path, mask_path, class_name = (
            self.image_paths[index], self.mask_paths[index], self.labels[index]
        )
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        original_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        semantic_mask = np.zeros_like(original_mask, dtype=np.int64)
        if class_name in ["benign", "malignant"]:
            semantic_mask[original_mask > 0] = 1
        return image, semantic_mask
class CVCDataset(Dataset):
    """
    PyTorch Dataset class for the CVC-ClinicDB dataset, specifically using PNG files.
    
    Args:
        root_dir (str): The root directory of the CVC-ClinicDB dataset.
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir

        # --- Paths are now hardcoded to use the PNG directories ---
        original_dir = os.path.join(self.root_dir, 'PNG', 'Original')
        mask_dir = os.path.join(self.root_dir, 'PNG', 'Ground Truth')

        # --- Find all .png images in the 'Original' folder ---
        self.image_paths = sorted(glob.glob(os.path.join(original_dir, '*.png')))
        self.mask_paths = []
        
        # Match images to their corresponding masks by filename
        for img_path in self.image_paths:
            file_name = os.path.basename(img_path)
            mask_path = os.path.join(mask_dir, file_name)
            if os.path.exists(mask_path):
                self.mask_paths.append(mask_path)
        
        assert len(self.image_paths) != 0, f"No images found in {original_dir}"
        assert len(self.image_paths) == len(self.mask_paths), "Mismatch in image and mask counts."

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        
        # Load the color image and convert from BGR to RGB
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load the mask in grayscale
        original_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Create a binary semantic mask (0 for background, 1 for polyp)
        semantic_mask = np.zeros_like(original_mask, dtype=np.int64)
        semantic_mask[original_mask > 0] = 1
        
        return image, semantic_mask
class TransformedSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            augmented = self.transform(image=x, mask=y)
            image = augmented['image']
            mask = augmented['mask']
        return image, mask.unsqueeze(0).float()
    
    def __len__(self):
        return len(self.subset)

def get_dataloaders(config):
    """Creates and returns the training and validation dataloaders."""
    # Define transforms
    root_dir = config['root_dir']
    img_height = config['dataloader_params']['img_height']
    img_width = config['dataloader_params']['img_width']
    mean = config['dataloader_params']['mean']
    std = config['dataloader_params']['std']
    batch_size = config['dataloader_params']['batch_size']
    seed = config['seed']

    
    
    # Create and split dataset
    if config['dataset_name'] == 'BUSI':
        full_dataset = BUSIDataset(root_dir=root_dir)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
    elif config['dataset_name'] == 'CVC-ClinicDB':
        full_dataset = CVCDataset(root_dir=root_dir)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
    else:
        raise ValueError(f"Dataset {config['dataset_name']} not supported.")
    g = torch.Generator()
    g.manual_seed(seed)
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=g)
    train_transform = A.Compose([
        A.RandomRotate90(),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Resize(img_height, img_width),
        A.Normalize(mean=mean.tolist(), std=std.tolist()),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(img_height, img_width),
        A.Normalize(mean=mean.tolist(), std=std.tolist()),
        ToTensorV2()
    ])
    # Apply transforms
    train_dataset = TransformedSubset(train_subset, transform=train_transform)
    val_dataset = TransformedSubset(val_subset, transform=val_transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,worker_init_fn=seed_worker,generator=g, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4,worker_init_fn=seed_worker, pin_memory=True)
    
    print(f"Full dataset size: {len(full_dataset)}")
    print(f"Found {len(train_dataset)} training images.")
    print(f"Found {len(val_dataset)} validation images.")
    
    return train_loader, val_loader