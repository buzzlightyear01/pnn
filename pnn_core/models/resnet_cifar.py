import torch.nn as nn
from torchvision.models import resnet18

class ResNet18CIFAR(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        model = resnet18(weights=None)

        # CIFAR adaptation
        model.conv1 = nn.Conv2d(
            3, 64,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        model.maxpool = nn.Identity()

        model.fc = nn.Linear(model.fc.in_features, num_classes)

        self.model = model

    def forward(self, x):
        return self.model(x)

