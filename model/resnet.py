# ResNet
# Deep Residual Learning for Image Recognition
# Kaiming He Xiangyu Zhang Shaoqing Ren Jian Sun

import os
import math
import torch
import torch.nn as nn
import torchvision.models
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

# you need to download the models to ~/.torch/models
# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
# }

models_dir = os.path.expanduser('checkpoints')
model_name = {
    'resnet18': 'resnet18-5c106cde.pth',
    'resnet34': 'resnet34-333f7ec4.pth',
    'resnet50': 'resnet50-19c8e357.pth',
    'resnet101': 'resnet101-5d3b4d8f.pth',
    'resnet152': 'resnet152-b121ed2d.pth',
    'resnetvgg': 'resnet50_face_sfew_dag.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        b,c,h,w = x.shape
        x = x.view(b,c,-1).permute(0,2,1)

        return x


def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet18'])))
    return model


def resnet34(pretrained=True, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet34'])))
    return model


def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet50'])))
    return model


def resnet101(pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet101'])))
    return model


def resnet152(pretrained=True, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet152'])))
    return model

class Resnet50_face_sfew_dag(nn.Module):

    def __init__(self):
        super(Resnet50_face_sfew_dag, self).__init__()
        self.meta = {'mean': [131.88330078125, 105.51170349121094, 92.56940460205078],
                     'std': [1, 1, 1],
                     'imageSize': [224, 224, 3, 32]}
        self.conv1_conv = nn.Conv2d(3, 64, kernel_size=[7, 7], stride=(2, 2), padding=(3, 3))
        self.conv1_bn = nn.BatchNorm2d(64, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv1_relu = nn.ReLU()
        self.conv1_pool = nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=1, dilation=1, ceil_mode=False)
        self.conv2_1a_conv = nn.Conv2d(64, 64, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv2_1a_bn = nn.BatchNorm2d(64, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_1a_relu = nn.ReLU()
        self.conv2_1b_conv = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv2_1b_bn = nn.BatchNorm2d(64, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_1b_relu = nn.ReLU()
        self.conv2_1c_conv = nn.Conv2d(64, 256, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv2_1c_bn = nn.BatchNorm2d(256, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_1_adapt_conv = nn.Conv2d(64, 256, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv2_1_adapt_bn = nn.BatchNorm2d(256, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_1_relu = nn.ReLU()
        self.conv2_2a_conv = nn.Conv2d(256, 64, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv2_2a_bn = nn.BatchNorm2d(64, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_2a_relu = nn.ReLU()
        self.conv2_2b_conv = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv2_2b_bn = nn.BatchNorm2d(64, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_2b_relu = nn.ReLU()
        self.conv2_2c_conv = nn.Conv2d(64, 256, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv2_2c_bn = nn.BatchNorm2d(256, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_2_relu = nn.ReLU()
        self.conv2_3a_conv = nn.Conv2d(256, 64, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv2_3a_bn = nn.BatchNorm2d(64, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_3a_relu = nn.ReLU()
        self.conv2_3b_conv = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv2_3b_bn = nn.BatchNorm2d(64, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_3b_relu = nn.ReLU()
        self.conv2_3c_conv = nn.Conv2d(64, 256, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv2_3c_bn = nn.BatchNorm2d(256, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_3_relu = nn.ReLU()
        self.conv3_1a_conv = nn.Conv2d(256, 128, kernel_size=[1, 1], stride=(2, 2), bias=False)
        self.conv3_1a_bn = nn.BatchNorm2d(128, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_1a_relu = nn.ReLU()
        self.conv3_1b_conv = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv3_1b_bn = nn.BatchNorm2d(128, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_1b_relu = nn.ReLU()
        self.conv3_1c_conv = nn.Conv2d(128, 512, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv3_1c_bn = nn.BatchNorm2d(512, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_1_adapt_conv = nn.Conv2d(256, 512, kernel_size=[1, 1], stride=(2, 2), bias=False)
        self.conv3_1_adapt_bn = nn.BatchNorm2d(512, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_1_relu = nn.ReLU()
        self.conv3_2a_conv = nn.Conv2d(512, 128, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv3_2a_bn = nn.BatchNorm2d(128, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_2a_relu = nn.ReLU()
        self.conv3_2b_conv = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv3_2b_bn = nn.BatchNorm2d(128, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_2b_relu = nn.ReLU()
        self.conv3_2c_conv = nn.Conv2d(128, 512, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv3_2c_bn = nn.BatchNorm2d(512, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_2_relu = nn.ReLU()
        self.conv3_3a_conv = nn.Conv2d(512, 128, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv3_3a_bn = nn.BatchNorm2d(128, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_3a_relu = nn.ReLU()
        self.conv3_3b_conv = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv3_3b_bn = nn.BatchNorm2d(128, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_3b_relu = nn.ReLU()
        self.conv3_3c_conv = nn.Conv2d(128, 512, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv3_3c_bn = nn.BatchNorm2d(512, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_3_relu = nn.ReLU()
        self.conv3_4a_conv = nn.Conv2d(512, 128, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv3_4a_bn = nn.BatchNorm2d(128, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_4a_relu = nn.ReLU()
        self.conv3_4b_conv = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv3_4b_bn = nn.BatchNorm2d(128, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_4b_relu = nn.ReLU()
        self.conv3_4c_conv = nn.Conv2d(128, 512, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv3_4c_bn = nn.BatchNorm2d(512, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_4_relu = nn.ReLU()
        self.conv4_1a_conv = nn.Conv2d(512, 256, kernel_size=[1, 1], stride=(2, 2), bias=False)
        self.conv4_1a_bn = nn.BatchNorm2d(256, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_1a_relu = nn.ReLU()
        self.conv4_1b_conv = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv4_1b_bn = nn.BatchNorm2d(256, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_1b_relu = nn.ReLU()
        self.conv4_1c_conv = nn.Conv2d(256, 1024, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv4_1c_bn = nn.BatchNorm2d(1024, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_1_adapt_conv = nn.Conv2d(512, 1024, kernel_size=[1, 1], stride=(2, 2), bias=False)
        self.conv4_1_adapt_bn = nn.BatchNorm2d(1024, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_1_relu = nn.ReLU()
        self.conv4_2a_conv = nn.Conv2d(1024, 256, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv4_2a_bn = nn.BatchNorm2d(256, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_2a_relu = nn.ReLU()
        self.conv4_2b_conv = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv4_2b_bn = nn.BatchNorm2d(256, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_2b_relu = nn.ReLU()
        self.conv4_2c_conv = nn.Conv2d(256, 1024, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv4_2c_bn = nn.BatchNorm2d(1024, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_2_relu = nn.ReLU()
        self.conv4_3a_conv = nn.Conv2d(1024, 256, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv4_3a_bn = nn.BatchNorm2d(256, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_3a_relu = nn.ReLU()
        self.conv4_3b_conv = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv4_3b_bn = nn.BatchNorm2d(256, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_3b_relu = nn.ReLU()
        self.conv4_3c_conv = nn.Conv2d(256, 1024, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv4_3c_bn = nn.BatchNorm2d(1024, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_3_relu = nn.ReLU()
        self.conv4_4a_conv = nn.Conv2d(1024, 256, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv4_4a_bn = nn.BatchNorm2d(256, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_4a_relu = nn.ReLU()
        self.conv4_4b_conv = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv4_4b_bn = nn.BatchNorm2d(256, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_4b_relu = nn.ReLU()
        self.conv4_4c_conv = nn.Conv2d(256, 1024, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv4_4c_bn = nn.BatchNorm2d(1024, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_4_relu = nn.ReLU()
        self.conv4_5a_conv = nn.Conv2d(1024, 256, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv4_5a_bn = nn.BatchNorm2d(256, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_5a_relu = nn.ReLU()
        self.conv4_5b_conv = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv4_5b_bn = nn.BatchNorm2d(256, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_5b_relu = nn.ReLU()
        self.conv4_5c_conv = nn.Conv2d(256, 1024, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv4_5c_bn = nn.BatchNorm2d(1024, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_5_relu = nn.ReLU()
        self.conv4_6a_conv = nn.Conv2d(1024, 256, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv4_6a_bn = nn.BatchNorm2d(256, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_6a_relu = nn.ReLU()
        self.conv4_6b_conv = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv4_6b_bn = nn.BatchNorm2d(256, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_6b_relu = nn.ReLU()
        self.conv4_6c_conv = nn.Conv2d(256, 1024, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv4_6c_bn = nn.BatchNorm2d(1024, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4_6_relu = nn.ReLU()
        self.conv5_1a_conv = nn.Conv2d(1024, 512, kernel_size=[1, 1], stride=(2, 2), bias=False)
        self.conv5_1a_bn = nn.BatchNorm2d(512, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5_1a_relu = nn.ReLU()
        self.conv5_1b_conv = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv5_1b_bn = nn.BatchNorm2d(512, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5_1b_relu = nn.ReLU()
        self.conv5_1c_conv = nn.Conv2d(512, 2048, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv5_1c_bn = nn.BatchNorm2d(2048, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5_1_adapt_conv = nn.Conv2d(1024, 2048, kernel_size=[1, 1], stride=(2, 2), bias=False)
        self.conv5_1_adapt_bn = nn.BatchNorm2d(2048, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5_1_relu = nn.ReLU()
        self.conv5_2a_conv = nn.Conv2d(2048, 512, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv5_2a_bn = nn.BatchNorm2d(512, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5_2a_relu = nn.ReLU()
        self.conv5_2b_conv = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv5_2b_bn = nn.BatchNorm2d(512, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5_2b_relu = nn.ReLU()
        self.conv5_2c_conv = nn.Conv2d(512, 2048, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv5_2c_bn = nn.BatchNorm2d(2048, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5_2_relu = nn.ReLU()
        self.conv5_3a_conv = nn.Conv2d(2048, 512, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv5_3a_bn = nn.BatchNorm2d(512, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5_3a_relu = nn.ReLU()
        self.conv5_3b_conv = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1), bias=False)
        self.conv5_3b_bn = nn.BatchNorm2d(512, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5_3b_relu = nn.ReLU()
        self.conv5_3c_conv = nn.Conv2d(512, 2048, kernel_size=[1, 1], stride=(1, 1), bias=False)
        self.conv5_3c_bn = nn.BatchNorm2d(2048, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5_3_relu = nn.ReLU()
        self.prediction_avg = nn.AvgPool2d(kernel_size=[7, 7], stride=[1, 1], padding=0)
        self.prediction = nn.Linear(in_features=2048, out_features=7, bias=True)

    def forward(self, data):
        conv1_conv = self.conv1_conv(data)
        conv1_bn = self.conv1_bn(conv1_conv)
        conv1_relu = self.conv1_relu(conv1_bn)
        conv1 = self.conv1_pool(conv1_relu)
        conv2_1a_conv = self.conv2_1a_conv(conv1)
        conv2_1a_bn = self.conv2_1a_bn(conv2_1a_conv)
        conv2_1a_relu = self.conv2_1a_relu(conv2_1a_bn)
        conv2_1b_conv = self.conv2_1b_conv(conv2_1a_relu)
        conv2_1b_bn = self.conv2_1b_bn(conv2_1b_conv)
        conv2_1b_relu = self.conv2_1b_relu(conv2_1b_bn)
        conv2_1c_conv = self.conv2_1c_conv(conv2_1b_relu)
        conv2_1c_bn = self.conv2_1c_bn(conv2_1c_conv)
        conv1_adapted = self.conv2_1_adapt_conv(conv1)
        conv1_adapted_bn = self.conv2_1_adapt_bn(conv1_adapted)
        conv2_1_sum = torch.add(conv1_adapted_bn, 1, conv2_1c_bn)
        conv2_1 = self.conv2_1_relu(conv2_1_sum)
        conv2_2a_conv = self.conv2_2a_conv(conv2_1)
        conv2_2a_bn = self.conv2_2a_bn(conv2_2a_conv)
        conv2_2a_relu = self.conv2_2a_relu(conv2_2a_bn)
        conv2_2b_conv = self.conv2_2b_conv(conv2_2a_relu)
        conv2_2b_bn = self.conv2_2b_bn(conv2_2b_conv)
        conv2_2b_relu = self.conv2_2b_relu(conv2_2b_bn)
        conv2_2c_conv = self.conv2_2c_conv(conv2_2b_relu)
        conv2_2c_bn = self.conv2_2c_bn(conv2_2c_conv)
        conv2_2_sum = torch.add(conv2_1, 1, conv2_2c_bn)
        conv2_2 = self.conv2_2_relu(conv2_2_sum)
        conv2_3a_conv = self.conv2_3a_conv(conv2_2)
        conv2_3a_bn = self.conv2_3a_bn(conv2_3a_conv)
        conv2_3a_relu = self.conv2_3a_relu(conv2_3a_bn)
        conv2_3b_conv = self.conv2_3b_conv(conv2_3a_relu)
        conv2_3b_bn = self.conv2_3b_bn(conv2_3b_conv)
        conv2_3b_relu = self.conv2_3b_relu(conv2_3b_bn)
        conv2_3c_conv = self.conv2_3c_conv(conv2_3b_relu)
        conv2_3c_bn = self.conv2_3c_bn(conv2_3c_conv)
        conv2_3_sum = torch.add(conv2_2, 1, conv2_3c_bn)
        conv2_3 = self.conv2_3_relu(conv2_3_sum)
        conv3_1a_conv = self.conv3_1a_conv(conv2_3)
        conv3_1a_bn = self.conv3_1a_bn(conv3_1a_conv)
        conv3_1a_relu = self.conv3_1a_relu(conv3_1a_bn)
        conv3_1b_conv = self.conv3_1b_conv(conv3_1a_relu)
        conv3_1b_bn = self.conv3_1b_bn(conv3_1b_conv)
        conv3_1b_relu = self.conv3_1b_relu(conv3_1b_bn)
        conv3_1c_conv = self.conv3_1c_conv(conv3_1b_relu)
        conv3_1c_bn = self.conv3_1c_bn(conv3_1c_conv)
        conv2_3_adapted = self.conv3_1_adapt_conv(conv2_3)
        conv2_3_adapted_bn = self.conv3_1_adapt_bn(conv2_3_adapted)
        conv3_1_sum = torch.add(conv2_3_adapted_bn, 1, conv3_1c_bn)
        conv3_1 = self.conv3_1_relu(conv3_1_sum)
        conv3_2a_conv = self.conv3_2a_conv(conv3_1)
        conv3_2a_bn = self.conv3_2a_bn(conv3_2a_conv)
        conv3_2a_relu = self.conv3_2a_relu(conv3_2a_bn)
        conv3_2b_conv = self.conv3_2b_conv(conv3_2a_relu)
        conv3_2b_bn = self.conv3_2b_bn(conv3_2b_conv)
        conv3_2b_relu = self.conv3_2b_relu(conv3_2b_bn)
        conv3_2c_conv = self.conv3_2c_conv(conv3_2b_relu)
        conv3_2c_bn = self.conv3_2c_bn(conv3_2c_conv)
        conv3_2_sum = torch.add(conv3_1, 1, conv3_2c_bn)
        conv3_2 = self.conv3_2_relu(conv3_2_sum)
        conv3_3a_conv = self.conv3_3a_conv(conv3_2)
        conv3_3a_bn = self.conv3_3a_bn(conv3_3a_conv)
        conv3_3a_relu = self.conv3_3a_relu(conv3_3a_bn)
        conv3_3b_conv = self.conv3_3b_conv(conv3_3a_relu)
        conv3_3b_bn = self.conv3_3b_bn(conv3_3b_conv)
        conv3_3b_relu = self.conv3_3b_relu(conv3_3b_bn)
        conv3_3c_conv = self.conv3_3c_conv(conv3_3b_relu)
        conv3_3c_bn = self.conv3_3c_bn(conv3_3c_conv)
        conv3_3_sum = torch.add(conv3_2, 1, conv3_3c_bn)
        conv3_3 = self.conv3_3_relu(conv3_3_sum)
        conv3_4a_conv = self.conv3_4a_conv(conv3_3)
        conv3_4a_bn = self.conv3_4a_bn(conv3_4a_conv)
        conv3_4a_relu = self.conv3_4a_relu(conv3_4a_bn)
        conv3_4b_conv = self.conv3_4b_conv(conv3_4a_relu)
        conv3_4b_bn = self.conv3_4b_bn(conv3_4b_conv)
        conv3_4b_relu = self.conv3_4b_relu(conv3_4b_bn)
        conv3_4c_conv = self.conv3_4c_conv(conv3_4b_relu)
        conv3_4c_bn = self.conv3_4c_bn(conv3_4c_conv)
        conv3_4_sum = torch.add(conv3_3, 1, conv3_4c_bn)
        conv3_4 = self.conv3_4_relu(conv3_4_sum)
        conv4_1a_conv = self.conv4_1a_conv(conv3_4)
        conv4_1a_bn = self.conv4_1a_bn(conv4_1a_conv)
        conv4_1a_relu = self.conv4_1a_relu(conv4_1a_bn)
        conv4_1b_conv = self.conv4_1b_conv(conv4_1a_relu)
        conv4_1b_bn = self.conv4_1b_bn(conv4_1b_conv)
        conv4_1b_relu = self.conv4_1b_relu(conv4_1b_bn)
        conv4_1c_conv = self.conv4_1c_conv(conv4_1b_relu)
        conv4_1c_bn = self.conv4_1c_bn(conv4_1c_conv)
        conv3_4_adapted = self.conv4_1_adapt_conv(conv3_4)
        conv3_4_adapted_bn = self.conv4_1_adapt_bn(conv3_4_adapted)
        conv4_1_sum = torch.add(conv3_4_adapted_bn, 1, conv4_1c_bn)
        conv4_1 = self.conv4_1_relu(conv4_1_sum)
        conv4_2a_conv = self.conv4_2a_conv(conv4_1)
        conv4_2a_bn = self.conv4_2a_bn(conv4_2a_conv)
        conv4_2a_relu = self.conv4_2a_relu(conv4_2a_bn)
        conv4_2b_conv = self.conv4_2b_conv(conv4_2a_relu)
        conv4_2b_bn = self.conv4_2b_bn(conv4_2b_conv)
        conv4_2b_relu = self.conv4_2b_relu(conv4_2b_bn)
        conv4_2c_conv = self.conv4_2c_conv(conv4_2b_relu)
        conv4_2c_bn = self.conv4_2c_bn(conv4_2c_conv)
        conv4_2_sum = torch.add(conv4_1, 1, conv4_2c_bn)
        conv4_2 = self.conv4_2_relu(conv4_2_sum)
        conv4_3a_conv = self.conv4_3a_conv(conv4_2)
        conv4_3a_bn = self.conv4_3a_bn(conv4_3a_conv)
        conv4_3a_relu = self.conv4_3a_relu(conv4_3a_bn)
        conv4_3b_conv = self.conv4_3b_conv(conv4_3a_relu)
        conv4_3b_bn = self.conv4_3b_bn(conv4_3b_conv)
        conv4_3b_relu = self.conv4_3b_relu(conv4_3b_bn)
        conv4_3c_conv = self.conv4_3c_conv(conv4_3b_relu)
        conv4_3c_bn = self.conv4_3c_bn(conv4_3c_conv)
        conv4_3_sum = torch.add(conv4_2, 1, conv4_3c_bn)
        conv4_3 = self.conv4_3_relu(conv4_3_sum)
        conv4_4a_conv = self.conv4_4a_conv(conv4_3)
        conv4_4a_bn = self.conv4_4a_bn(conv4_4a_conv)
        conv4_4a_relu = self.conv4_4a_relu(conv4_4a_bn)
        conv4_4b_conv = self.conv4_4b_conv(conv4_4a_relu)
        conv4_4b_bn = self.conv4_4b_bn(conv4_4b_conv)
        conv4_4b_relu = self.conv4_4b_relu(conv4_4b_bn)
        conv4_4c_conv = self.conv4_4c_conv(conv4_4b_relu)
        conv4_4c_bn = self.conv4_4c_bn(conv4_4c_conv)
        conv4_4_sum = torch.add(conv4_3, 1, conv4_4c_bn)
        conv4_4 = self.conv4_4_relu(conv4_4_sum)
        conv4_5a_conv = self.conv4_5a_conv(conv4_4)
        conv4_5a_bn = self.conv4_5a_bn(conv4_5a_conv)
        conv4_5a_relu = self.conv4_5a_relu(conv4_5a_bn)
        conv4_5b_conv = self.conv4_5b_conv(conv4_5a_relu)
        conv4_5b_bn = self.conv4_5b_bn(conv4_5b_conv)
        conv4_5b_relu = self.conv4_5b_relu(conv4_5b_bn)
        conv4_5c_conv = self.conv4_5c_conv(conv4_5b_relu)
        conv4_5c_bn = self.conv4_5c_bn(conv4_5c_conv)
        conv4_5_sum = torch.add(conv4_4, 1, conv4_5c_bn)
        conv4_5 = self.conv4_5_relu(conv4_5_sum)
        conv4_6a_conv = self.conv4_6a_conv(conv4_5)
        conv4_6a_bn = self.conv4_6a_bn(conv4_6a_conv)
        conv4_6a_relu = self.conv4_6a_relu(conv4_6a_bn)
        conv4_6b_conv = self.conv4_6b_conv(conv4_6a_relu)
        conv4_6b_bn = self.conv4_6b_bn(conv4_6b_conv)
        conv4_6b_relu = self.conv4_6b_relu(conv4_6b_bn)
        conv4_6c_conv = self.conv4_6c_conv(conv4_6b_relu)
        conv4_6c_bn = self.conv4_6c_bn(conv4_6c_conv)
        conv4_6_sum = torch.add(conv4_5, 1, conv4_6c_bn)
        conv4_6 = self.conv4_6_relu(conv4_6_sum)
        conv5_1a_conv = self.conv5_1a_conv(conv4_6)
        conv5_1a_bn = self.conv5_1a_bn(conv5_1a_conv)
        conv5_1a_relu = self.conv5_1a_relu(conv5_1a_bn)
        conv5_1b_conv = self.conv5_1b_conv(conv5_1a_relu)
        conv5_1b_bn = self.conv5_1b_bn(conv5_1b_conv)
        conv5_1b_relu = self.conv5_1b_relu(conv5_1b_bn)
        conv5_1c_conv = self.conv5_1c_conv(conv5_1b_relu)
        conv5_1c_bn = self.conv5_1c_bn(conv5_1c_conv)
        conv4_6_adapted = self.conv5_1_adapt_conv(conv4_6)
        conv4_6_adapted_bn = self.conv5_1_adapt_bn(conv4_6_adapted)
        conv5_1_sum = torch.add(conv4_6_adapted_bn, 1, conv5_1c_bn)
        conv5_1 = self.conv5_1_relu(conv5_1_sum)
        conv5_2a_conv = self.conv5_2a_conv(conv5_1)
        conv5_2a_bn = self.conv5_2a_bn(conv5_2a_conv)
        conv5_2a_relu = self.conv5_2a_relu(conv5_2a_bn)
        conv5_2b_conv = self.conv5_2b_conv(conv5_2a_relu)
        conv5_2b_bn = self.conv5_2b_bn(conv5_2b_conv)
        conv5_2b_relu = self.conv5_2b_relu(conv5_2b_bn)
        conv5_2c_conv = self.conv5_2c_conv(conv5_2b_relu)
        conv5_2c_bn = self.conv5_2c_bn(conv5_2c_conv)
        conv5_2_sum = torch.add(conv5_1, 1, conv5_2c_bn)
        conv5_2 = self.conv5_2_relu(conv5_2_sum)
        conv5_3a_conv = self.conv5_3a_conv(conv5_2)
        conv5_3a_bn = self.conv5_3a_bn(conv5_3a_conv)
        conv5_3a_relu = self.conv5_3a_relu(conv5_3a_bn)
        conv5_3b_conv = self.conv5_3b_conv(conv5_3a_relu)
        conv5_3b_bn = self.conv5_3b_bn(conv5_3b_conv)
        conv5_3b_relu = self.conv5_3b_relu(conv5_3b_bn)
        conv5_3c_conv = self.conv5_3c_conv(conv5_3b_relu)
        conv5_3c_bn = self.conv5_3c_bn(conv5_3c_conv)
        conv5_3_sum = torch.add(conv5_2, 1, conv5_3c_bn)
        conv5_3 = self.conv5_3_relu(conv5_3_sum)
        # prediction_avg_preflatten = self.prediction_avg(conv5_3)
        # prediction_avg = prediction_avg_preflatten.view(prediction_avg_preflatten.size(0), -1)
        # prediction = self.prediction(prediction_avg)
        # return prediction
        return conv5_3

def resnet50_face_sfew_dag(weights_path=None, **kwargs):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    model = Resnet50_face_sfew_dag()
    if weights_path:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnetvgg'])))
    return model
