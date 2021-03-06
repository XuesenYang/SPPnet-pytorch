import math
import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import torch.nn.functional as F
import torchvision
from spatial_pyramid_pool import spatial_pyramid_pool


def calc_auto(lst, channels):
    return sum(map(lambda x: x ** 2, lst)) * channels


class SPP_NET(torch.nn.Module):
    '''
    A VGG-16 model which adds spp layer so that we can input multi-size tensor
    '''

    def __init__(self,
                 number_classes: int = 2):
        super(SPP_NET, self).__init__()

        self.features = torchvision.models.vgg16(pretrained=False)
        self.number_classes = number_classes
        self.features = torch.nn.Sequential(*list(self.features.children())[:-1])
        self.output_num = [4, 2, 1]
        self.spp = spatial_pyramid_pool(self.output_num)
        fmsize_list = []
        for name, m in self.features.named_modules():
            if isinstance(m, torch.nn.Conv2d):
                fmsize_list.append(m.out_channels)
        self.num_ftrs = fmsize_list[-1] # last conv kenel size

        self.fc1 = nn.Linear(calc_auto(self.output_num, self.num_ftrs), self.number_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.spp(x)
        x = self.fc1(x)

        return x
