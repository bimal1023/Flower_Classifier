import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

def get_model(num_classes, drop_out=0.4, pretrained=True):
    if pretrained:
        weights = ResNet18_Weights.DEFAULT
    else:
        weights = None

    model = models.resnet18(weights=weights)

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=drop_out),
        nn.Linear(num_features, num_classes)
    )

    return model