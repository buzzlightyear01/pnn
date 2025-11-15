import torch.nn as nn
from torchvision.models import resnet18

def build_resnet18(num_classes: int = 100, pretrained: bool = False) -> nn.Module:
    """
    ResNet-18 برای CIFAR-100.
    اگر pretrained=True باشد، وزن‌های ImageNet لود می‌شوند (اگر نسخه‌ی torchvision پشتیبانی کند).
    """
    try:
        # برای نسخه‌های جدید torchvision (weights)
        from torchvision.models import ResNet18_Weights
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
    except Exception:
        # برای نسخه‌های قدیمی‌تر (pretrained)
        model = resnet18(pretrained=pretrained)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
