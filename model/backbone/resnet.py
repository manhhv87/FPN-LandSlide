import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']

model_urls = {
    'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
    'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
    'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=1,
                     stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, down_sample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(in_planes=in_planes, out_planes=planes)
        self.bn1 = nn.BatchNorm2d(num_features=planes)
        self.conv2 = conv3x3(in_planes=planes, out_planes=planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(num_features=planes)
        self.conv3 = conv1x1(in_planes=planes, out_planes=planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(num_features=planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample is not None:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, pretrained=True):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # Modules
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block=block, planes=64, blocks=layers[0])
        self.layer2 = self._make_layer(block=block, planes=128, blocks=layers[1], stride=2)
        self.layer3 = self._make_layer(block=block, planes=256, blocks=layers[2], stride=2)
        self.layer4 = self._make_layer(block=block, planes=512, blocks=layers[3], stride=2)
        self._init_weights()

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1):
        down_sample = None

        if stride != 1 or self.in_planes != planes * block.expansion:
            down_sample = nn.Sequential(
                conv1x1(in_planes=self.in_planes, out_planes=planes * block.expansion, stride=stride),
                nn.BatchNorm2d(num_features=planes * block.expansion),
            )

        layers = [block(self.in_planes, planes, stride, down_sample)]
        self.in_planes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        low_level_feat1 = x

        x = self.layer1(x)
        low_level_feat2 = x
        x = self.layer2(x)
        low_level_feat3 = x
        x = self.layer3(x)
        low_level_feat4 = x
        x = self.layer4(x)
        low_level_feat5 = x
        return [low_level_feat1, low_level_feat2, low_level_feat3, low_level_feat4, low_level_feat5]

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
        model_dict = {}
        state_dict = self.state_dict()

        for k, v in pretrained_dict.items():
            if k in state_dict:
                model_dict[k] = v

        state_dict.update(model_dict)

        self.load_state_dict(state_dict)


def ResNet18(pretrained=True):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pretrained on ImageNet
    """
    model = ResNet(Bottleneck, [2, 2, 2, 2])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def ResNet34(pretrained=False):
    """Constructs a ResNet-34 model.
    Args:
      pretrained (bool): If True, returns a model pretrained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def ResNet50(pretrained=True):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pretrained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def ResNet101(pretrained=True):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pretrained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def ResNet152(pretrained=True):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pretrained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
