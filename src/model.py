import torch
import torch.nn as nn
from torchvision import models

def get_model(architecture: str = "resnet18", num_classes: int = 10) -> nn.Module:
    if architecture == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")