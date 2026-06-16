import torch.nn as nn
from torchvision import models


def build_model(device):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 128),
        nn.ReLU(),
        nn.Linear(128, 3),
    )

    return model.to(device)
