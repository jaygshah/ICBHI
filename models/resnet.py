import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck


class ResNet2D(nn.Module):
    def __init__(self, block, layers, in_channels=3, num_classes=1000):
        super(ResNet2D, self).__init__()
        self.inplanes = 64

        # Initial Convolutional Layer
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # MaxPooling Layer
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet Layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Global Average Pooling and Fully Connected Layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize Weights
        self._initialize_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        """Create a ResNet layer."""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = torch.sigmoid(self.fc(x))

        return x


def resnet18_2d(in_channels=3, num_classes=1000):
    """Construct a ResNet-18 model."""
    return ResNet2D(BasicBlock, [2, 2, 2, 2], in_channels, num_classes)


def resnet34_2d(in_channels=3, num_classes=1000):
    """Construct a ResNet-34 model."""
    return ResNet2D(BasicBlock, [3, 4, 6, 3], in_channels, num_classes)


def resnet50_2d(in_channels=3, num_classes=1000):
    """Construct a ResNet-50 model."""
    return ResNet2D(Bottleneck, [3, 4, 6, 3], in_channels, num_classes)


def resnet101_2d(in_channels=3, num_classes=1000):
    """Construct a ResNet-101 model."""
    return ResNet2D(Bottleneck, [3, 4, 23, 3], in_channels, num_classes)


def resnet152_2d(in_channels=3, num_classes=1000):
    """Construct a ResNet-152 model."""
    return ResNet2D(Bottleneck, [3, 8, 36, 3], in_channels, num_classes)
