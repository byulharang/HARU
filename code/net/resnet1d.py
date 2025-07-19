import torch
import torch.nn as nn
import math

__all__ = ['ResNet1D', 'resnet1d18', 'resnet1d34', 'resnet1d50', 'resnet101', 'resnet152']

def conv1d3x3(in_planes, out_planes, stride=1):
    """1D 3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def get_gn1d(channels, num_groups=32):
    """1D GroupNorm for given channel count"""
    return nn.GroupNorm(num_groups=min(num_groups, channels), num_channels=channels)

class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout_prob=0.0):
        super().__init__()
        self.conv1 = conv1d3x3(inplanes, planes, stride)
        self.gn1   = get_gn1d(planes)
        self.relu  = nn.ReLU(inplace=True)

        self.conv2 = conv1d3x3(planes, planes)
        self.gn2   = get_gn1d(planes)

        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob>0 else nn.Identity()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return self.dropout(out)

class Bottleneck1D(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout_prob=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.gn1   = get_gn1d(planes)
        self.conv2 = conv1d3x3(planes, planes, stride)
        self.gn2   = get_gn1d(planes)
        self.conv3 = nn.Conv1d(planes, planes*4, kernel_size=1, bias=False)
        self.gn3   = get_gn1d(planes*4)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob>0 else nn.Identity()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.gn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return self.dropout(out)

class ResNet1D(nn.Module):
    def __init__(self, block, layers, dropout_prob=0.1):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv1d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.gn1   = get_gn1d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 128, layers[0], stride=1, dropout_prob=dropout_prob)
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2, dropout_prob=dropout_prob)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2, dropout_prob=dropout_prob)
        self.layer4 = self._make_layer(block, 1024, layers[3], stride=2, dropout_prob=dropout_prob)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                k = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size,)
                n = math.prod(k) * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def _make_layer(self, block, planes, blocks, stride=1, dropout_prob=0.0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                get_gn1d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dropout_prob))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dropout_prob=dropout_prob))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x  # B x 1024 x L'

# Factory functions

def resnet1d18(pretrained=False, **kwargs):
    return ResNet1D(BasicBlock1D, [2, 2, 2, 2], **kwargs)

def resnet1d34(pretrained=False, **kwargs):
    return ResNet1D(BasicBlock1D, [3, 4, 6, 3], **kwargs)

def resnet1d50(pretrained=False, **kwargs):
    return ResNet1D(Bottleneck1D, [3, 4, 6, 3], **kwargs)

def resnet101(pretrained=False, **kwargs):
    return ResNet1D(Bottleneck1D, [3, 4, 23, 3], **kwargs)

def resnet152(pretrained=False, **kwargs):
    return ResNet1D(Bottleneck1D, [3, 8, 36, 3], **kwargs)
