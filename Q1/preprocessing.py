import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torchvision.transforms as transforms

class ImagePreprocessor:
    def __init__(self, target_size=(288, 288)):
        self.target_size = target_size
    
    def resize_image(self, image):
        if isinstance(image, np.ndarray):
            return cv2.resize(image, self.target_size)
        elif isinstance(image, Image.Image):
            return image.resize(self.target_size)
        return image
    
    def normalize_image(self, image):
        if isinstance(image, np.ndarray):
            return image.astype(np.float32) / 255.0
        elif isinstance(image, torch.Tensor):
            return image.float() / 255.0
        return image
    
    def enhance_contrast(self, image, factor=1.5):
        if isinstance(image, Image.Image):
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(factor)
        return image
    
    def enhance_brightness(self, image, factor=1.2):
        if isinstance(image, Image.Image):
            enhancer = ImageEnhance.Brightness(image)
            return enhancer.enhance(factor)
        return image
    
    def remove_noise(self, image):
        if isinstance(image, np.ndarray):
            return cv2.medianBlur(image, 3)
        elif isinstance(image, Image.Image):
            return image.filter(ImageFilter.MedianFilter(size=3))
        return image
    
    def sharpen_image(self, image):
        if isinstance(image, Image.Image):
            return image.filter(ImageFilter.SHARPEN)
        return image
    
    def adjust_hsv(self, image, h_shift=0, s_scale=1.0, v_scale=1.0):
        if isinstance(image, np.ndarray):
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            
            if h_shift != 0:
                h = (h.astype(int) + h_shift) % 180
            if s_scale != 1.0:
                s = np.clip(s * s_scale, 0, 255)
            if v_scale != 1.0:
                v = np.clip(v * v_scale, 0, 255)
            
            hsv = cv2.merge([h, s, v])
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return image
    
    def crop_center(self, image, crop_ratio=0.9):
        if isinstance(image, Image.Image):
            width, height = image.size
            new_width = int(width * crop_ratio)
            new_height = int(height * crop_ratio)
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            right = (width + new_width) // 2
            bottom = (height + new_height) // 2
            return image.crop((left, top, right, bottom))
        return image
    
    def preprocess_for_training(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        image = self.resize_image(image)
        image = self.crop_center(image)
        image = self.enhance_contrast(image)
        image = self.enhance_brightness(image)
        
        return image
    
    def preprocess_for_inference(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        image = self.resize_image(image)
        image = self.sharpen_image(image)
        
        return image

def get_train_transforms(target_size=(288, 288)):
    return transforms.Compose([
        transforms.Lambda(lambda x: ImagePreprocessor(target_size).preprocess_for_training(x)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
        transforms.RandomResizedCrop(target_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms(target_size=(288, 288)):
    return transforms.Compose([
        transforms.Lambda(lambda x: ImagePreprocessor(target_size).preprocess_for_inference(x)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])