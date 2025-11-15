import torch.nn as nn
from typing import List


class CNN(nn.Module):
    def __init__(self, input_channels: int, num_classes: int, channels: List[int] = None, dropout: float = 0.0):
        super().__init__()
        if channels is None:
            channels = [32, 64]
        
        layers = []
        prev_channels = input_channels
        
        for ch in channels:
            layers.append(nn.Conv2d(prev_channels, ch, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            prev_channels = ch
        
        self.features = nn.Sequential(*layers)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prev_channels * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

