import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

from augmentation import get_train_transforms, get_val_transforms
from config import Config


class CropDiseaseDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self._load_from_directory()
    
    def _load_from_directory(self):
        images_dir = os.path.join(self.data_dir, 'images')
        if not os.path.exists(images_dir):
            print(f"Images directory not found: {images_dir}")
            return
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        for filename in os.listdir(images_dir):
            if os.path.splitext(filename)[1].lower() in image_extensions:
                img_path = os.path.join(images_dir, filename)
                class_id = int(filename.split('_')[0])
                self.samples.append((img_path, class_id))
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_data_loaders(config):
    train_transform = get_train_transforms(
        img_size=config.IMG_SIZE,
        crop_scale=config.CROP_SCALE,
        crop_ratio=config.CROP_RATIO,
        color_jitter=config.COLOR_JITTER,
        grayscale_prob=config.GRAYSCALE_PROB,
        erasing_prob=config.ERASING_PROB,
        randaugment_num_ops=config.RANDAUGMENT_NUM_OPS,
        randaugment_magnitude=config.RANDAUGMENT_MAGNITUDE
    )
    
    val_transform = get_val_transforms(img_size=config.IMG_SIZE)
    
    train_dataset = CropDiseaseDataset(config.DATA_DIR, transform=train_transform)
    val_dataset = CropDiseaseDataset(config.VAL_DIR, transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, len(train_dataset), len(val_dataset)


def analyze_class_distribution(data_dir):
    class_counts = {}
    
    images_dir = os.path.join(data_dir, 'images')
    if os.path.exists(images_dir):
        for filename in os.listdir(images_dir):
            if os.path.splitext(filename)[1].lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}:
                try:
                    class_id = int(filename.split('_')[0])
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
                except (ValueError, IndexError):
                    print(f"Warning: Could not extract class from filename: {filename}")
    
    sorted_counts = sorted(class_counts.items())
    
    return {
        'class_counts': dict(sorted_counts),
        'total_samples': sum(class_counts.values()),
        'num_classes': len(class_counts),
        'min_samples': min(class_counts.values()) if class_counts else 0,
        'max_samples': max(class_counts.values()) if class_counts else 0
    }