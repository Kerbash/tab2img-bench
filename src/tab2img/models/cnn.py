"""CNN classifiers for T2I images (PyTorch).

All take (batch, 3, H, W) input and work for any H, W >= 1. The output layer
has 1 unit for binary problems (sigmoid / BCE) and k units otherwise.

simple     the benchmark CNN: 2 conv layers + adaptive 4x4 pool + dense (LeNet-like)
deep       2 double-conv blocks with max-pooling + 2 dense layers
resnet18   ImageNet-pretrained ResNet-18, image upsampled to 64x64
linear     flatten -> linear (logistic regression on the pixels)
tiny_gap   one 3x3 conv (16 filters) -> global average pool -> linear
tiny_flat  one 3x3 conv (4 filters) -> flatten -> linear (position-aware)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _out(num_classes):
    return 1 if num_classes == 2 else num_classes


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, _out(num_classes))

    def forward(self, x):
        x = self.pool(F.relu(self.c2(F.relu(self.c1(x)))))
        return self.fc2(self.drop(F.relu(self.fc1(torch.flatten(x, 1)))))


class DeepCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
                                nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.b2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                                nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.4),
                                nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4),
                                nn.Linear(128, _out(num_classes)))

    def forward(self, x):
        x = self.b1(x)
        if x.shape[2] >= 2 and x.shape[3] >= 2:
            x = F.max_pool2d(x, 2)
        x = self.b2(x)
        if x.shape[2] >= 2 and x.shape[3] >= 2:
            x = F.max_pool2d(x, 2)
        return self.fc(torch.flatten(self.pool(x), 1))


class PretrainedResNet(nn.Module):
    def __init__(self, num_classes, size=64):
        super().__init__()
        import torchvision.models as tvm
        self.net = tvm.resnet18(weights="IMAGENET1K_V1")
        self.net.fc = nn.Linear(512, _out(num_classes))
        self.size = size
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        return self.net((x - self.mean) / self.std)


class LinearPix(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc = nn.LazyLinear(_out(num_classes))

    def forward(self, x):
        return self.fc(torch.flatten(x, 1))


class TinyGAP(nn.Module):
    def __init__(self, num_classes, k=16):
        super().__init__()
        self.c = nn.Conv2d(3, k, 3, padding=1)
        self.fc = nn.Linear(k, _out(num_classes))

    def forward(self, x):
        return self.fc(F.relu(self.c(x)).mean(dim=(2, 3)))


class TinyFlat(nn.Module):
    def __init__(self, num_classes, k=4):
        super().__init__()
        self.c = nn.Conv2d(3, k, 3, padding=1)
        self.fc = nn.LazyLinear(_out(num_classes))

    def forward(self, x):
        return self.fc(torch.flatten(F.relu(self.c(x)), 1))


ARCHITECTURES = {"simple": SimpleCNN, "deep": DeepCNN, "resnet18": PretrainedResNet,
                 "linear": LinearPix, "tiny_gap": TinyGAP, "tiny_flat": TinyFlat}
