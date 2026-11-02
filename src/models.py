"""A custom CNN baseline and ImageNet-pretrained EfficientNet-B0 fine-tuning."""
import torch
from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class BaselineCNN(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()
        blocks=[]; inp=3
        for out in (32,64,128,192):
            blocks += [nn.Conv2d(inp,out,3,padding=1,bias=False),nn.BatchNorm2d(out),
                       nn.ReLU(inplace=True),nn.MaxPool2d(2)]
            inp=out
        self.features=nn.Sequential(*blocks)
        self.classifier=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),
                                      nn.Dropout(0.3),nn.Linear(192,num_classes))
    def forward(self,x): return self.classifier(self.features(x))

def build_model(name='efficientnet',num_classes=10,pretrained=True,freeze_backbone=False):
    if name=='baseline':return BaselineCNN(num_classes)
    if name!='efficientnet':raise ValueError(name)
    weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model=efficientnet_b0(weights=weights)
    if freeze_backbone:
        for p in model.features.parameters():p.requires_grad=False
    in_features=model.classifier[-1].in_features
    model.classifier=nn.Sequential(nn.Dropout(0.3),nn.Linear(in_features,num_classes))
    return model

def last_conv(model):
    if isinstance(model,BaselineCNN):return model.features[12]
    # EfficientNet's last 1x1 conv -> spatial feature tensor
    return model.features[-1][0]

def count_params(model): return sum(p.numel() for p in model.parameters())
