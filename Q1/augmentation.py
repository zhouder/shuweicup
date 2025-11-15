import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import torch.nn.functional as F


class RandomResizedCrop(transforms.RandomResizedCrop):
    def __init__(self, size, scale=(0.7, 1.0), ratio=(3/4, 4/3)):
        super().__init__(size, scale=scale, ratio=ratio)


class RandomErasing(object):
    def __init__(self, probability=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0):
        self.probability = probability
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def __call__(self, img):
        if random.random() > self.probability:
            return img
            
        if isinstance(img, torch.Tensor):
            if random.random() > self.probability:
                return img
            for _ in range(100):
                area = img.shape[1] * img.shape[2]
                target_area = random.uniform(*self.scale) * area
                aspect_ratio = random.uniform(*self.ratio)

                h = int(round((target_area * aspect_ratio) ** 0.5))
                w = int(round((target_area / aspect_ratio) ** 0.5))

                if w < img.shape[2] and h < img.shape[1]:
                    x1 = random.randint(0, img.shape[1] - h)
                    y1 = random.randint(0, img.shape[2] - w)
                    if img.shape[0] == 3:
                        img[0, x1:x1+h, y1:y1+w] = self.value
                        img[1, x1:x1+h, y1:y1+w] = self.value
                        img[2, x1:x1+h, y1:y1+w] = self.value
                    else:
                        img[0, x1:x1+h, y1:y1+w] = self.value
                    return img
            return img
        else:
            return transforms.RandomErasing(probability=self.probability, scale=self.scale, 
                                          ratio=self.ratio, value=self.value)(img)


class RandAugment:
    def __init__(self, num_ops=2, magnitude=8):
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.augment_list = [
            (self.auto_contrast, 0, 1),
            (self.equalize, 0, 1),
            (self.invert, 0, 1),
            (self.rotate, 0, 30),
            (self.posterize, 0, 4),
            (self.solarize, 0, 256),
            (self.solarize_add, 0, 110),
            (self.color, 0.1, 1.9),
            (self.contrast, 0.1, 1.9),
            (self.brightness, 0.1, 1.9),
            (self.sharpness, 0.1, 1.9),
            (self.shear_x, 0., 0.3),
            (self.shear_y, 0., 0.3),
            (self.translate_x_rel, 0., 0.45),
            (self.translate_y_rel, 0., 0.45),
        ]

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.num_ops)
        for op, minval, maxval in ops:
            val = (float(self.magnitude) / 30) * float(maxval - minval) + minval
            img = op(img, val)
        return img

    def auto_contrast(self, pil_img, level):
        return ImageOps.autocontrast(pil_img)

    def equalize(self, pil_img, level):
        return ImageOps.equalize(pil_img)

    def invert(self, pil_img, level):
        return ImageOps.invert(pil_img)

    def rotate(self, pil_img, level):
        degrees = int_parameter(level, 30)
        if random.random() > 0.5:
            degrees = -degrees
        return pil_img.rotate(degrees, resample=Image.BILINEAR)

    def posterize(self, pil_img, level):
        level = int_parameter(level, 4)
        return ImageOps.posterize(pil_img, 4 - level)

    def solarize(self, pil_img, level):
        level = int_parameter(level, 256)
        return ImageOps.solarize(pil_img, 256 - level)

    def solarize_add(self, pil_img, level):
        level = int_parameter(level, 110)
        if random.random() > 0.5:
            level = -level
        img_np = np.array(pil_img).astype(np.int)
        img_np = img_np + level
        img_np = np.clip(img_np, 0, 255)
        return Image.fromarray(img_np.astype(np.uint8))

    def color(self, pil_img, level):
        level = float_parameter(level, 1.8) + 0.1
        return ImageEnhance.Color(pil_img).enhance(level)

    def contrast(self, pil_img, level):
        level = float_parameter(level, 1.8) + 0.1
        return ImageEnhance.Contrast(pil_img).enhance(level)

    def brightness(self, pil_img, level):
        level = float_parameter(level, 1.8) + 0.1
        return ImageEnhance.Brightness(pil_img).enhance(level)

    def sharpness(self, pil_img, level):
        level = float_parameter(level, 1.8) + 0.1
        return ImageEnhance.Sharpness(pil_img).enhance(level)

    def shear_x(self, pil_img, level):
        level = float_parameter(level, 0.3)
        if random.random() > 0.5:
            level = -level
        return pil_img.transform(pil_img.size, Image.AFFINE, (1, level, 0, 0, 1, 0), resample=Image.BILINEAR)

    def shear_y(self, pil_img, level):
        level = float_parameter(level, 0.3)
        if random.random() > 0.5:
            level = -level
        return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, level, 1, 0), resample=Image.BILINEAR)

    def translate_x_rel(self, pil_img, level):
        level = float_parameter(level, 0.45)
        if random.random() > 0.5:
            level = -level
        level = int(level * pil_img.size[0])
        return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, level, 0, 1, 0), resample=Image.BILINEAR)

    def translate_y_rel(self, pil_img, level):
        level = float_parameter(level, 0.45)
        if random.random() > 0.5:
            level = -level
        level = int(level * pil_img.size[1])
        return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, 0, 1, level), resample=Image.BILINEAR)


def int_parameter(level, maxval):
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    return float(level) * maxval / 10.


try:
    from PIL import ImageOps
except ImportError:
    ImageOps = None


def get_train_transforms(img_size=384, crop_scale=(0.7, 1.0), crop_ratio=(3/4, 4/3), 
                       color_jitter=None, grayscale_prob=0.1, erasing_prob=0.25,
                       randaugment_num_ops=2, randaugment_magnitude=8):
    if color_jitter is None:
        color_jitter = {
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.2,
            'hue': 0.1
        }
    
    return transforms.Compose([
        RandomResizedCrop(img_size, scale=crop_scale, ratio=crop_ratio),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(**color_jitter),
        transforms.RandomGrayscale(p=grayscale_prob),
        RandAugment(num_ops=randaugment_num_ops, magnitude=randaugment_magnitude),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        RandomErasing(probability=erasing_prob)
    ])


def get_val_transforms(img_size=384):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])