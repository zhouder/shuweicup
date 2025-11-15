import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

from config import Config

class CropDiseaseDataset(Dataset):
    def __init__(self, data_dir, list_file=None, json_file=None, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        if list_file:
            self._load_from_list_file(list_file)
        elif json_file:
            self._load_from_json_file(json_file)
        else:
            self._load_from_directory()
    
    def _load_from_list_file(self, list_file):
        with open(list_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_path = parts[0]
                    label = int(parts[1])
                    self.samples.append((img_path, label))
    
    def _load_from_json_file(self, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
            for item in data:
                img_path = os.path.join(self.data_dir, 'images', item['image_id'])
                label = item['disease_class']
                self.samples.append((img_path, label))
    
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

def _build_train_transform():
    jitter_cfg = Config.COLOR_JITTER
    return transforms.Compose([
        transforms.RandomResizedCrop(
            Config.IMG_SIZE,
            scale=Config.CROP_SCALE,
            ratio=Config.CROP_RATIO
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=jitter_cfg['brightness'],
            contrast=jitter_cfg['contrast'],
            saturation=jitter_cfg['saturation'],
            hue=jitter_cfg['hue']
        ),
        transforms.RandAugment(
            num_ops=Config.RANDAUGMENT_NUM_OPS,
            magnitude=Config.RANDAUGMENT_MAGNITUDE
        ),
        transforms.RandomGrayscale(p=Config.GRAYSCALE_PROB),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(
            p=Config.ERASING_PROB,
            scale=(0.02, 0.15),
            ratio=(0.3, 3.3)
        ),
    ])

def _build_val_transform():
    return transforms.Compose([
        transforms.Resize(Config.IMG_SIZE + 32),
        transforms.CenterCrop(Config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_data_loaders(train_dir, val_dir, batch_size=32, num_workers=4):
    train_transform = _build_train_transform()
    val_transform = _build_val_transform()
    
    train_dataset = CropDiseaseDataset(train_dir, transform=train_transform)
    val_dataset = CropDiseaseDataset(val_dir, transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, len(train_dataset), len(val_dataset)
