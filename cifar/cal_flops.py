from __future__ import print_function

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import os
import shutil
import argparse
import time
import logging

import models_baseline as models
from data import *
import torch.nn.functional as F
from torch.autograd import Variable
from data import *
from util_adv import wrm, fgm, ifgm, ifgm_branch, ifgm_max, ifgm_max_v2, ifgm_k, ifgm_max_v4
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith('__')
                     and callable(models.__dict__[name]))
count_ops = 0
num_ids = 0                   

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 training')
    # hyper-parameters are from ResNet paper
    parser.add_argument('arch', metavar='ARCH', default='cifar10_resnet_110',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: cifar10_resnet_110)')

    args = parser.parse_args()
    return args



def get_feature_hook(self, _input, _output):
    global count_ops, num_ids
    print('------>>>>>>')
    print('{}th node, input shape: {}, output shape: {}, input channel: {}, output channel {}'.format(
        num_ids, _input[0].size(2), _output.size(2), _input[0].size(1), _output.size(1)))
    print(self)
    delta_ops = self.in_channels * self.out_channels * self.kernel_size[0] * self.kernel_size[1] * _output.size(2) * _output.size(3) / self.groups
    count_ops += delta_ops
    print('ops is {:.6f}M'.format(delta_ops / 1024.  /1024.))
    num_ids += 1
    print('')

def measure_model(net, H_in, W_in):
    '''
    Args:
       net: pytorch network, father class is nn.Module
       H_in: int, input image height
       W_in: int, input image weight
    '''
    _input = Variable(torch.randn((1, 3, H_in, W_in)))
    #_input, net = _input.cpu(), net.cpu()
    hooks = []
    for module in net.named_modules():
        if isinstance(module[1], nn.Conv2d) or isinstance(module[1], nn.ConvTranspose2d) :
            print(module)
            hooks.append(module[1].register_forward_hook(get_feature_hook))
    _out = net(_input)
    global count_ops
    print('count_ops: {:.6f}M'.format(count_ops / 1024. /1024.)) # in Million (edited) 


def main():
    args = parse_args()
    model = models.__dict__[args.arch](False)
    #model = torch.nn.DataParallel(model).cuda()
    measure_model(model, 32, 32)






if __name__ == '__main__':
    main()
