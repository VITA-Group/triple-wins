"""prepare MNIST
"""

from __future__ import print_function

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

padding = 4


def prepare_train_data(dataset='mnist', batch_size=128,
                       shuffle=True, num_workers=4):

    if 'MNIST' in dataset:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

        trainset = torchvision.datasets.__dict__[dataset.upper()](
            root='./data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=batch_size,
                                                   shuffle=shuffle,
                                                   num_workers=num_workers)
    else:
        train_loader = None

    return train_loader


def prepare_test_data(dataset='MNIST', batch_size=128,
                      shuffle=False, num_workers=4):

    if 'MNIST' in dataset:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

        testset = torchvision.datasets.__dict__[dataset.upper()](root='./data',
                                               train=False,
                                               download=True,
                                               transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers)

    else:
        test_loader = None
    return test_loader
