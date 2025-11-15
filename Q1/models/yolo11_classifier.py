import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError(
        "需要安装 ultralytics：pip install ultralytics\n"
        f"Import ultralytics 失败：{e}"
    )


def _find_last_linear(model: nn.Module):
    last_name, last_mod = None, None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            last_name, last_mod = name, mod
    if last_mod is None:
        raise RuntimeError("未在 YOLO11-cls 模型中找到 Linear 分类层。")
    return last_name, last_mod


def _set_module(parent: nn.Module, name: str, new_mod: nn.Module):
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
        self.model = yolo_model.model
        
        last_name, last_mod = _find_last_linear(self.model)
        in_features = last_mod.in_features
        new_fc = nn.Linear(in_features, num_classes)
        nn.init.normal_(new_fc.weight, std=0.01)
        nn.init.zeros_(new_fc.bias)
        _set_module(self.model, last_name, new_fc)
        
        self.num_classes = num_classes
        
    def forward(self, x):
        return self.model(x)
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            if len(x.shape) == 3:
                x = x.unsqueeze(0)
            outputs = self.forward(x)
            _, preds = torch.max(outputs, 1)
            return preds
    
    def get_features(self, x):
        feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        with torch.no_grad():
            features = feature_extractor(x)
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = torch.flatten(features, 1)
        return features