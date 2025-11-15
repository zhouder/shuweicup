import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18Baseline(nn.Module):
    def __init__(self, num_classes=61, pretrained=True):
        super(ResNet18Baseline, self).__init__()
        
        self.model = models.resnet18(pretrained=pretrained)
        
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
        nn.init.normal_(self.model.fc.weight, std=0.01)
        nn.init.zeros_(self.model.fc.bias)
        
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
            features = torch.flatten(features, 1)
        return features