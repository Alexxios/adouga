"""ResNet18-based classification model."""

import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class GameClassifier(nn.Module):
    """ResNet18-based binary classifier."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
