""" This file contains the model definitions for both original ResNet (6n+2
layers) and SkipNets.
"""

import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F


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
        self.layers = layers


        # group 0
        downsample = None
        stride = 1
        planes = 16
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        setattr(self, 'group0_layer%s' % 0, block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, layers[0]):
            setattr(self, 'group0_layer%s' % i, block(self.inplanes, planes))
 
        # group 1
        downsample = None
        stride = 2
        planes = 32
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        setattr(self, 'group1_layer%s' % 0, block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, layers[1]):
            setattr(self, 'group1_layer%s' % i, block(self.inplanes, planes))
 

        # group 2
        downsample = None
        stride = 2
        planes = 64
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        setattr(self, 'group2_layer%s' % 0, block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, layers[2]):
            setattr(self, 'group2_layer%s' % i, block(self.inplanes, planes))
 


        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)



        ## branch network 1
        branch_channels = [16, 64, 32]
        self.branch_layer1 = nn.Sequential(
                              nn.MaxPool2d(2, stride=2),
                              nn.Conv2d(branch_channels[0], 
                              branch_channels[1], kernel_size=5,
                              stride=1, padding=2,
                              bias=False),
                              BatchNorm(branch_channels[1]),
                              nn.ReLU(inplace=True),
                              nn.MaxPool2d(2, stride=2),
                              nn.Conv2d(branch_channels[1], branch_channels[2],
                              kernel_size=3,stride=1, padding=1,
                              bias=False),
                              BatchNorm(branch_channels[2]),
                              nn.AvgPool2d(8),
                              View(-1, branch_channels[2]),
                              nn.Linear(32 * block.expansion, num_classes)
                              )

        ## branch network 2
        branch_channels = [16, 64, 32]
        self.branch_layer2 = nn.Sequential(
                              nn.MaxPool2d(2, stride=2),
                              nn.Conv2d(branch_channels[0], 
                              branch_channels[1], kernel_size=5,
                              stride=1, padding=2,
                              bias=False),
                              BatchNorm(branch_channels[1]),
                              nn.ReLU(inplace=True),
                              nn.MaxPool2d(2, stride=2),
                              nn.Conv2d(branch_channels[1], branch_channels[2],
                              kernel_size=3,stride=1, padding=1,
                              bias=False),
                              BatchNorm(branch_channels[2]),
                              nn.AvgPool2d(8),
                              View(-1, branch_channels[2]),
                              nn.Linear(32 * block.expansion, num_classes)
                              )



        ## branch network 3
        branch_channels = [32, 32]
        self.branch_layer3 = nn.Sequential(
                              nn.MaxPool2d(2, stride=2),
                              block(branch_channels[0], branch_channels[1] * block.expansion, 1, None),
                              nn.AvgPool2d(8),
                              View(-1, branch_channels[1]),
                              nn.Linear(32 * block.expansion, num_classes)
                              )

        ## branch network 4
        branch_channels = [32, 32]
        self.branch_layer4 = nn.Sequential(
                              nn.MaxPool2d(2, stride=2),
                              block(branch_channels[0], branch_channels[1] * block.expansion, 1, None),
                              nn.AvgPool2d(8),
                              View(-1, branch_channels[1]),
                              nn.Linear(32 * block.expansion, num_classes)
                              )


        ## branch network 5
        self.branch_layer5 = nn.Sequential(
                              nn.AvgPool2d(8),
                              View(-1, 64 * block.expansion),
                              nn.Linear(64 * block.expansion, num_classes)
                              )

        ## branch network 6
        self.branch_layer6 = nn.Sequential(
                              nn.AvgPool2d(8),
                              View(-1, 64 * block.expansion),
                              nn.Linear(64 * block.expansion, num_classes)
                              )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        output_branch = []
        for g in range(3):
            for i in range(self.layers[g]):
                x = getattr(self, 'group{}_layer{}'.format(g, i))(x)
                if i == int( self.layers[g] * 1/4) and g == 0 :
                    x_branch1 = self.branch_layer1(x)
                    output_branch.append(x_branch1)
                if i == int( self.layers[g] * 3/4) and g == 0 :
                    x_branch2 = self.branch_layer2(x)
                    output_branch.append(x_branch2)
                if i == int( self.layers[g] * 1/4) and g == 1 :
                    x_branch3 = self.branch_layer3(x)
                    output_branch.append(x_branch3)
                if i == int( self.layers[g] * 3/4) and g == 1 :
                    x_branch4 = self.branch_layer4(x)
                    output_branch.append(x_branch4)
                if i == int( self.layers[g] * 1/4) and g == 2 :
                    x_branch5 = self.branch_layer5(x)
                    output_branch.append(x_branch5)
                if i == int( self.layers[g] * 3/4) and g == 2 :
                    x_branch6 = self.branch_layer6(x)
                    output_branch.append(x_branch6)



        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        output_branch.append(x)
        return output_branch

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


        ## branch network 1
        branch_channels = [16, 32, 64]
        branch_channels = [int(self.alpha * x) for x in branch_channels]
        self.branch_layer1 = nn.Sequential(
                              Block_V2(branch_channels[0], 
                              branch_channels[1], expansion,
                              stride=2),
                              BatchNorm(branch_channels[1]),
                              nn.ReLU(inplace=True),
                              Block_V2(branch_channels[1], branch_channels[2],
                              expansion, stride=2),
                              BatchNorm(branch_channels[2]),
                              nn.AvgPool2d(8),
                              View(-1, branch_channels[2]),
                              nn.Linear(branch_channels[-1], num_classes)
                              )



        ## branch network 2
        branch_channels = [32, 32]
        branch_channels = [int(self.alpha * x) for x in branch_channels]
        self.branch_layer2 = nn.Sequential(
                              Block_V2(branch_channels[0], 
                              branch_channels[1], expansion,
                              stride=2),
                              nn.AvgPool2d(8),
                              View(-1, branch_channels[1]),
                              nn.Linear(branch_channels[-1], num_classes)
                              )

        ## branch network 3
        self.branch_layer3 = nn.Sequential(
                              nn.AvgPool2d(8),
                              View(-1, int(self.alpha * 64)),
                              nn.Linear(int(self.alpha * 64), num_classes)
                              )


        ## branch network 4
        self.branch_layer4 = nn.Sequential(
                              nn.AvgPool2d(8),
                              View(-1, int(self.alpha * 64)),
                              nn.Linear(int(self.alpha * 64), num_classes)
                              )


        ## branch network 5
        self.branch_layer5 = nn.Sequential(
                              nn.AvgPool2d(8),
                              View(-1, int(self.alpha * 96)),
                              nn.Linear(int(self.alpha * 96), num_classes)
                              )


        ## branch network 6
        self.branch_layer6 = nn.Sequential(
                              nn.AvgPool2d(4),
                              View(-1, int(self.alpha * 160)),
                              nn.Linear(int(self.alpha * 160), num_classes)
                              )


    def forward(self, x):
        
        out = F.relu(self.bn1(self.layer1(x)))
        output_branch = []
        for layer_idx in range(2, self.layer_num):
            out = getattr(self, 'layer{}'.format(layer_idx))(out)
            if layer_idx == int( self.layer_num * 1/7):
               x = self.branch_layer1(out)
               output_branch.append(x)
            if layer_idx == int( self.layer_num * 2/7):
               x = self.branch_layer2(out)
               output_branch.append(x)
            if layer_idx == int( self.layer_num * 3/7):
               x = self.branch_layer3(out)
               output_branch.append(x)
            if layer_idx == int( self.layer_num * 4/7):
               x = self.branch_layer4(out)
               output_branch.append(x)
            if layer_idx == int( self.layer_num * 5/7):
               x = self.branch_layer5(out)
               output_branch.append(x)
            if layer_idx == int( self.layer_num * 6/7):
               x = self.branch_layer6(out) 
               output_branch.append(x)
        out = F.relu(self.bn2(getattr(self, 'layer{}'.format(self.layer_num))(out)))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        output_branch.append(out)
        return output_branch


def cifar10_mobilenetv2_1(pretrained=False, **kwargs):
    model = MobileNetV2(alpha=1)
    return model


def cifar10_mobilenetv2_075(pretrained=False, **kwargs):
    model = MobileNetV2(alpha=0.75)
    return model




def main():
    print('test model')
    x = torch.rand(2, 3, 32, 32)
    model = cifar10_mobilenetv2_075()
    a = model(x)
    print(len(a))

if __name__ == "__main__":
    main()

