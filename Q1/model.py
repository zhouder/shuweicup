import torch
import torch.nn as nn
from ultralytics import YOLO
from torchvision.models import resnet18, ResNet18_Weights

def _find_last_linear(model):
    last_name, last_mod = None, None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            last_name, last_mod = name, mod
    if last_mod is None:
        raise RuntimeError("未在 YOLO11-cls 模型中找到 Linear 分类层。")
    return last_name, last_mod


def _set_module(parent, name, new_mod):
    parts = name.split(".")
    obj = parent
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], new_mod)


class YOLO11Classifier(nn.Module):
    def __init__(self, num_classes=61, pretrained=True):
        super(YOLO11Classifier, self).__init__()
        
        weights_name = "yolo11x-cls.pt"
        
        yolo_model = YOLO(weights_name)
        self.model = yolo_model.model  # 取出 nn.Module
        
        last_name, last_mod = _find_last_linear(self.model)
        in_features = last_mod.in_features
        new_fc = nn.Linear(in_features, num_classes)
        nn.init.normal_(new_fc.weight, std=0.01)
        nn.init.zeros_(new_fc.bias)
        _set_module(self.model, last_name, new_fc)
        
        self.num_classes = num_classes
        
    def forward(self, x):
        """前向传播"""
        return self.model(x)
    
    def predict(self, x):
        """预测方法"""
        self.eval()
        with torch.no_grad():
            if len(x.shape) == 3:  # 如果是单张图片
                x = x.unsqueeze(0)
            outputs = self.forward(x)
            _, preds = torch.max(outputs, 1)
            return preds

class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=61, pretrained=True):
        super(ResNet18Classifier, self).__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.model = resnet18(weights=weights)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        nn.init.trunc_normal_(self.model.fc.weight, std=0.02)
        nn.init.zeros_(self.model.fc.bias)
    
    def forward(self, x):
        return self.model(x)

def create_model(model_type='yolo11x-cls', num_classes=61, pretrained=True):
    if model_type == 'yolo11x-cls':
        return YOLO11Classifier(num_classes, pretrained)
    if model_type == 'resnet18':
        return ResNet18Classifier(num_classes, pretrained)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_model_info(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024)
    }
