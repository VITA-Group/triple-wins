""" This file contains the model definitions for both original ResNet (6n+2
layers) and SkipNets.
"""

import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np

BatchNorm = nn.BatchNorm2d
class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(*self.shape)





def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


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


########################################
# Original ResNet                      #
########################################


class ResNet(nn.Module):
    """Original ResNet without routing modules"""
    def __init__(self, block, layers, num_classes=10):
        self.inplanes = 16
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

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
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
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

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# For CIFAR-10
# ResNet-38
def cifar10_resnet_38(pretrained=False, **kwargs):
    # n = 6
    model = ResNet(BasicBlock, [6, 6, 6], **kwargs)
    return model


# ResNet-74
def cifar10_resnet_74(pretrained=False, **kwargs):
    # n = 12
    model = ResNet(BasicBlock, [12, 12, 12], **kwargs)
    return model


# ResNet-110
def cifar10_resnet_110(pretrained=False, **kwargs):
    # n = 18
    model = ResNet(BasicBlock, [18, 18, 18], **kwargs)
    return model


# ResNet-152
def cifar10_resnet_152(pretrained=False, **kwargs):
    # n = 25
    model = ResNet(BasicBlock, [25, 25, 25], **kwargs)
    return model



class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out



class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=10, alpha=1):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.alpha = alpha
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(int(1024 * alpha),  num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            out_planes =  int(out_planes * self.alpha)
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(int(in_planes), int(out_planes), stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def cifar10_mobilenet_1(pretrained=False, **kwargs):
    model = MobileNet(alpha=1)
    return model

def cifar10_mobilenet_075(pretrained=False, **kwargs):
    model = MobileNet(alpha=0.75)
    return model


def cifar10_mobilenet_05(pretrained=False, **kwargs):
    model = MobileNet(alpha=0.5)
    return model

def cifar10_mobilenet_025(pretrained=False, **kwargs):
    model = MobileNet(alpha=0.25)
    return model



def cifar10_mobilenet_01(pretrained=False, **kwargs):
    model = MobileNet(alpha=0.1)
    return model



class Block_V2(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block_V2, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10, alpha=1):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.alpha = alpha
        self.layer1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        layer_index = 2
        in_planes = 32
        for expansion, out_planes, num_blocks, stride in self.cfg:
            out_planes = int(self.alpha * out_planes)
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                setattr(self, 'layer%s' % layer_index, Block_V2(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
                layer_index += 1

        setattr(self, 'layer%s' % layer_index, nn.Conv2d(int(320 * self.alpha), int(1280 * self.alpha), kernel_size=1, stride=1, padding=0, bias=False))
        self.conv2 = nn.Conv2d(int(320 * self.alpha), int(1280 * self.alpha), kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(int(1280 * self.alpha))
        self.linear = nn.Linear(int(1280 * self.alpha), num_classes)
        self.layer_num = layer_index


    def forward(self, x):
        
        out = F.relu(self.bn1(self.layer1(x)))
        output_branch = []
        for layer_idx in range(2, self.layer_num):
            out = getattr(self, 'layer{}'.format(layer_idx))(out)
        out = F.relu(self.bn2(getattr(self, 'layer{}'.format(self.layer_num))(out)))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



def cifar10_mobilenetv2_1(pretrained=False, **kwargs):
    model = MobileNetV2(alpha=1)
    return model

def cifar10_mobilenetv2_09(pretrained=False, **kwargs):
    model = MobileNetV2(alpha=0.9)
    return model

def cifar10_mobilenetv2_075(pretrained=False, **kwargs):
    model = MobileNetV2(alpha=0.75)
    return model


def cifar10_mobilenetv2_05(pretrained=False, **kwargs):
    model = MobileNetV2(alpha=0.5)
    return model


def main():
    print('test model')
    x = torch.rand(2, 3, 32, 32)
    model = cifar10_mobilenetv2_1()
    a = model(x)
    print(a.shape)


class ChannelScalerLayer(nn.Module):
    ''' Scaler Layer for channels
    '''
    def __init__(self, channel_num):
        super(ChannelScalerLayer, self).__init__()
        self.channel_num = channel_num
        weight = np.array([1]*channel_num).reshape(channel_num, 1, 1)
        self.weight = nn.Parameter(torch.FloatTensor(weight))
    def __call__(self, x):
        # print(x.size(), self.weight.size())
        return self.weight * x

class BlockScalerLayer(nn.Module):
    ''' Scaler Layer for channels
    '''
    def __init__(self):
        super(BlockScalerLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor([1]))
    def __call__(self, x):
        # print(x.size(), self.weight.size())
        return self.weight * x



class Block_V3(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, mid_planes, stride):
        super(Block_V3, self).__init__()
        self.stride = stride

        # assert len(mid_planes) == 2

        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.cs1 = ChannelScalerLayer(mid_planes)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=mid_planes, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        
        if stride == 1:
            if in_planes == out_planes:
                self.shortcut = nn.Sequential()
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(out_planes),
                )                


    def forward(self, x):
        out = F.relu(self.cs1(self.bn1(self.conv1(x))))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.stride==1:
            out = out + self.shortcut(x)  
        else:
            out
        return out

class MobileNetV2_CP(nn.Module):

    def __init__(self, cfg=None, num_classes=10, alpha=1):
        super(MobileNetV2_CP, self).__init__()

        # 
        if cfg is None:
            # (expansion, out_planes, stride)
            cfg = [
                (32,  16, 1), # num_blocks=1

                (16*6,  24, 1), # num_blocks=2
                (24*6,  24, 1),

                (24*6,  32, 2), # num_blocks=3
                (32*6,  32, 1),
                (32*6,  32, 1),

                (32*6, 64, 2), # num_blocks=4
                (64*6, 64, 1),
                (64*6, 64, 1),
                (64*6, 64, 1),

                (64*6, 96, 1), # num_blocks=3
                (96*6, 96, 1),
                (96*6, 96, 1),

                (96*6, 160, 2), # num_blocks=3
                (160*6, 160, 1),
                (160*6, 160, 1),

                (160*6, 320, 1)]       
        self.cfg = cfg 


        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.alpha = alpha
        self.layer1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        layer_index = 2
        in_planes = 32
        for i, (mid_planes, out_planes, stride) in enumerate(self.cfg):
            out_planes = int(self.alpha * out_planes)
            if i != 0:
                mid_planes = int(mid_planes * self.alpha)
            setattr(self, 'layer%s' % layer_index, Block_V3(in_planes, out_planes, mid_planes, stride))
            in_planes = out_planes
            layer_index += 1

        setattr(self, 'layer%s' % layer_index, nn.Conv2d(int(320 * self.alpha), int(1280 * self.alpha), kernel_size=1, stride=1, padding=0, bias=False))
        self.bn2 = nn.BatchNorm2d(int(1280 * self.alpha))
        self.linear = nn.Linear(int(1280 * self.alpha), num_classes)
        self.layer_num = layer_index


    def forward(self, x):
        
        out = F.relu(self.bn1(self.layer1(x)))
        for layer_idx in range(2, self.layer_num):
            out = getattr(self, 'layer{}'.format(layer_idx))(out)
        out = F.relu(self.bn2(getattr(self, 'layer{}'.format(self.layer_num))(out)))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def cifar10_mobilenetv2_1_r001(pretrained=False, **kwargs):
    cfg = [[12, 16, 1],
          [54, 24, 1],
          [81, 24, 1],
          [125, 32, 2],
          [93, 32, 1],
          [93, 32, 1],
          [148, 64, 2],
          [167, 64, 1],
          [158, 64, 1],
          [156, 64, 1],
          [260, 96, 1],
          [168, 96, 1],
          [185, 96, 1],
          [326, 160, 2],
          [135, 160, 1],
          [63, 160, 1],
          [204, 320, 1]]
    model = MobileNetV2_CP(alpha=1, cfg=cfg)
    return model


def cifar10_mobilenetv2_1_r0009(pretrained=False, **kwargs):
    cfg = [[15, 16, 1],
          [56, 24, 1],
          [83, 24, 1],
          [124, 32, 2],
          [104, 32, 1],
          [96, 32, 1],
          [146, 64, 2],
          [153, 64, 1],
          [151, 64, 1],
          [135, 64, 1],
          [268, 96, 1],
          [208, 96, 1],
          [196, 96, 1],
          [373, 160, 2],
          [184, 160, 1],
          [142, 160, 1],
          [314, 320, 1]]

    model = MobileNetV2_CP(alpha=1, cfg=cfg)
    return model


def cifar10_mobilenetv2_1_r0008(pretrained=False, **kwargs):
    cfg = [[24, 16, 1],
          [78, 24, 1],
          [105, 24, 1],
          [122, 32, 2],
          [120, 32, 1],
          [96, 32, 1],
          [154, 64, 2],
          [222, 64, 1],
          [196, 64, 1],
          [205, 64, 1],
          [302, 96, 1],
          [217, 96, 1],
          [214, 96, 1],
          [345, 160, 2],
          [157, 160, 1],
          [65, 160, 1],
          [224, 320, 1]]

    model = MobileNetV2_CP(alpha=1, cfg=cfg)
    return model

def cifar10_mobilenetv2_1_r0007(pretrained=False, **kwargs):
    cfg = [[ 21,  16,   1],
            [ 70,  24,   1],
            [ 99,  24,   1],
            [129,  32,   2],
            [142,  32,   1],
            [125,  32,   1],
            [153,  64,   2],
            [232,  64,   1],
            [215,  64,   1],
            [195,  64,  1],
            [311,  96,   1],
            [278,  96,   1],
            [270,  96,   1],
            [386, 160,   2],
            [215, 160,   1],
            [115, 160,   1],
            [295, 320,   1]]

    model = MobileNetV2_CP(alpha=1, cfg=cfg)
    return model

def cifar10_mobilenetv2_1_r0006(pretrained=False, **kwargs):
    cfg = [[10, 16, 1],
          [67, 24, 1],
          [106, 24, 1],
          [139, 32, 2],
          [135, 32, 1],
          [140, 32, 1],
          [154, 64, 2],
          [230, 64, 1],
          [220, 64, 1],
          [213, 64, 1],
          [317, 96, 1],
          [313, 96, 1],
          [302, 96, 1],
          [436, 160, 2],
          [342, 160, 1],
          [268, 160, 1],
          [430, 320, 1]]

    model = MobileNetV2_CP(alpha=1, cfg=cfg)
    return model

def cifar10_mobilenetv2_1_r0005(pretrained=False, **kwargs):
    cfg = [[ 21,  16,   1],
            [ 81,  24,   1],
            [122,  24,   1],
            [137,  32,   2],
            [148,  32,   1],
            [143,  32,   1],
            [164,  64,   2],
            [291,  64,   1],
            [289,  64,   1],
            [273,  64,   1],
            [336,  96,   1],
            [398,  96,   1],
            [359,  96,   1],
            [435, 160,   2],
            [285, 160,   1],
            [184, 160,   1],
            [376, 320,   1]]

    model = MobileNetV2_CP(alpha=1, cfg=cfg)
    return model

def cifar10_mobilenetv2_1_r0004(pretrained=False, **kwargs):
    cfg = [[ 26,  16,   1],
            [ 90,  24,   1],
            [126,  24,   1],
            [142,  32,   2],
            [177,  32,   1],
            [171,  32,   1],
            [180,  64,   2],
            [336,  64,   1],
            [329,  64,   1],
            [342,  64,   1],
            [356,  96,   1],
            [422,  96,   1],
            [390,  96,   1],
            [485, 160,   2],
            [455, 160,   1],
            [293, 160,   1],
            [485, 320,   1]]

    model = MobileNetV2_CP(alpha=1, cfg=cfg)
    return model

def cifar10_mobilenetv2_1_r0003(pretrained=False, **kwargs):
    cfg = [[ 27,  16,   1],
            [ 90,  24,  1],
            [138,  24,   1],
            [142,  32,   2],
            [180,  32,   1],
            [184,  32,   1],
            [178,  64,   2],
            [354,  64,   1],
            [360,  64,   1],
            [346,  64,   1],
            [365,  96,   1],
            [499,  96,   1],
            [487,  96,   1],
            [515, 160,   2],
            [630, 160,   1],
            [491, 160,   1],
            [609, 320,   1]]

    model = MobileNetV2_CP(alpha=1, cfg=cfg)
    return model

def cifar10_mobilenetv2_1_r0002(pretrained=False, **kwargs):
    cfg = [[ 25,  16,  1],
            [ 91,  24,   1],
            [139,  24,   1],
            [143,  32,   2],
            [189,  32,   1],
            [185,  32,   1],
            [189,  64,   2],
            [371,  64,   1],
            [378,  64,   1],
            [370,  64,   1],
            [360,  96,   1],
            [557,  96,   1],
            [555,  96,   1],
            [549, 160,   2],
            [885, 160,   1],
            [812, 160,   1],
            [833, 320,   1]]

    model = MobileNetV2_CP(alpha=1, cfg=cfg)
    return model

class BasicBlockCP(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super(BasicBlockCP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.cs1 = ChannelScalerLayer(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # define residual conv:
        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))
        else:
            self.residual = nn.Sequential()

    def forward(self, x):
        out = self.relu(self.cs1(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))

        res_connect = self.residual(x)

        out += res_connect
        out = self.relu(out)
        return out


class ResNetCP(nn.Module):
    """Original ResNet without routing modules"""
    def __init__(self, cfg=None, num_classes=10):
        if cfg is None:
            cfg = [
                [16]*6,
                [32]*6,
                [64]*6,
            ]
        super(ResNetCP, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(16, cfg[0], 16)
        self.layer2 = self._make_layer(16, cfg[1], 32, stride=2)
        self.layer3 = self._make_layer(32, cfg[2], 64, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

        # weight initialization:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, in_channels, mid_channels_lst, out_channels, stride=1):
        block_num = len(mid_channels_lst)

        layers = []
        layers.append(BasicBlockCP(in_channels, mid_channels_lst[0], out_channels, stride=stride))
        for i in range(1, block_num):
            layers.append(BasicBlockCP(out_channels, mid_channels_lst[i], out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def rn38_r004(pretrained=False):
    cfg = [[15, 16, 16, 13, 16, 14], [32, 32, 31, 31, 27, 25], [64, 63, 59, 34, 20, 51]]
    return ResNetCP(cfg=cfg)

def rn38_r005(pretrained=False):
    cfg = [[15, 12, 13, 13, 15, 10], [32, 28, 27, 18, 13, 17], [62, 62, 42, 32, 25, 50]]
    return ResNetCP(cfg=cfg)

def rn38_r006(pretrained=False):
    cfg = [[16, 15, 10, 10, 11, 9], [30, 25, 15, 10, 13, 12], [56, 53, 36, 33, 13, 51]]
    return ResNetCP(cfg=cfg)

def rn38_r007(pretrained=False):
    cfg = [[13, 12, 12, 13, 10, 7], [32, 25, 21, 17, 12, 13], [61, 53, 31, 21, 18, 32]]
    return ResNetCP(cfg=cfg)

def rn38_r008(pretrained=False):
    cfg = [[12, 13, 9, 7, 12, 6], [31, 23, 20, 12, 10, 7], [56, 43, 26, 10, 19, 29]]
    return ResNetCP(cfg=cfg)


def main():
    x = torch.rand(16, 3, 32, 32)
    model = cifar10_mobilenetv2_1_r0006() 
    a = model(x)
    print(a.shape)


if __name__ == "__main__":
    main()
