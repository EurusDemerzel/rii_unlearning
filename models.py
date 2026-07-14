"""
Model definitions for the machine unlearning pipeline.

Supported architectures:
  - SimpleMLP:  2-layer MLP for MNIST  (784 → 128 → 10)
  - DeepMLP:    3-layer MLP for MNIST  (784 → 512 → 256 → 10)
  - SmallCNN:   3-conv + 2-fc CNN for CIFAR-10
"""

import torch
import torch.nn as nn


# =============================================================================
# MLP — MNIST
# =============================================================================
class SimpleMLP(nn.Module):
    """2-layer MLP: Flatten(784) → Linear(128) → ReLU → Linear(10)."""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 128,
                 num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# =============================================================================
# DeepMLP — MNIST (3-layer, for overfitting experiments)
# =============================================================================
class DeepMLP(nn.Module):
    """3-layer MLP: 784 → 512 → 256 → 10 (~535K params vs ~102K for SimpleMLP)."""

    def __init__(self, input_dim: int = 784, hidden1: int = 512,
                 hidden2: int = 256, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# =============================================================================
# CNN — CIFAR-10
# =============================================================================
class SmallCNN(nn.Module):
    """
    Small CNN for CIFAR-10:
      3 conv layers (32→64→128 filters) + 2 FC layers (256→10).
      BatchNorm + Dropout for regularisation.
    """

    def __init__(self, input_channels: int = 3, num_classes: int = 10):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        self.pool   = nn.MaxPool2d(2, 2)
        self.relu   = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        # After 3× pooling: 32 → 16 → 8 → 4  → 128 × 4 × 4 = 2048
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))          # 32×32 → 32×32
        x = self.pool(x)                                 # → 16×16
        x = self.relu(self.bn2(self.conv2(x)))          # 16×16
        x = self.pool(x)                                 # → 8×8
        x = self.relu(self.bn3(self.conv3(x)))          # 8×8
        x = self.pool(x)                                 # → 4×4
        x = torch.flatten(x, 1)                          # flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# =============================================================================
# ResNet-18  — CIFAR-100 scale experiments
# =============================================================================
class ResNet18CIFAR(nn.Module):
    """ResNet-18 adapted for CIFAR (32×32 input, no initial 7×7 conv).
    
    Uses BasicBlock with 64→128→256→512 channels, 
    total ~11M parameters. Suitable for CIFAR-10/100.
    """

    class BasicBlock(nn.Module):
        expansion = 1
        def __init__(self, in_planes, planes, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes))

        def forward(self, x):
            out = torch.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            return torch.relu(out)

    def __init__(self, num_classes=100):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(self.BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)


# =============================================================================
# Factory
# =============================================================================
_MODEL_REGISTRY = {
    "mlp":     SimpleMLP, "mlp2":    SimpleMLP,
    "mlp3":    DeepMLP,   "deepmlp": DeepMLP,
    "cnn":     SmallCNN,
    "resnet18": ResNet18CIFAR, "resnet": ResNet18CIFAR,
}

def get_model(dataset: str, device: torch.device, model_name: str = "mlp") -> nn.Module:
    """Return the appropriate model for the given dataset, moved to device.

    Args:
        dataset: 'mnist', 'cifar10', or 'cifar100'
        device: torch device
        model_name: 'mlp'/'mlp2' (SimpleMLP), 'mlp3'/'deepmlp' (DeepMLP),
                    'cnn' (SmallCNN), 'resnet18'/'resnet' (ResNet18CIFAR)
    """
    model_cls = _MODEL_REGISTRY.get(model_name.lower())
    if model_cls is None:
        raise ValueError(f"Unknown model: {model_name}. Options: {list(_MODEL_REGISTRY.keys())}")

    if dataset == "mnist":
        if model_cls in (SmallCNN, ResNet18CIFAR):
            raise ValueError(f"{model_name} not supported for MNIST. Use mlp or mlp3.")
        model = model_cls(input_dim=784, num_classes=10)
    elif dataset == "cifar10":
        model = model_cls(num_classes=10)
    elif dataset == "cifar100":
        model = model_cls(num_classes=100)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Supported: mnist, cifar10, cifar100")
    return model.to(device)


def clone_model(model: nn.Module, dataset: str, device: torch.device,
                model_name: str = "mlp") -> nn.Module:
    """Deep-copy a model (via state_dict)."""
    import copy
    new_model = get_model(dataset, device, model_name)
    new_model.load_state_dict(copy.deepcopy(model.state_dict()))
    return new_model
