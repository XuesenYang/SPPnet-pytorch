import math
import torch
import torch.nn as nn
import torchvision


class spatial_pyramid_pool(torch.nn.Module):
    """
    out_pool_size: a int vector of expected output size of max pooling layer
    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    """
    def __init__(self, out_pool_size, pooltype='max_pool'):
        super(spatial_pyramid_pool, self).__init__()
        self.out_pool_size = out_pool_size
        self.pooltype = pooltype

    def forward(self, x):
        num, c, h, w = x.size()
        for i in range(len(self.out_pool_size)):
            # 依据输入的feature map和 期望输出的值个数 调整pooling的尺寸 以及padding的尺寸
            h_wid = int(torch.ceil(torch.true_divide(h,  self.out_pool_size[i])))
            w_wid = int(torch.ceil(torch.true_divide(w,  self.out_pool_size[i])))
            h_pad = int((h_wid * self.out_pool_size[i] - h + 1) // 2)
            w_pad = int((w_wid * self.out_pool_size[i] - w + 1) // 2)
            maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
            x1 = maxpool(x)
            if (i == 0):
                spp = x1.view(num, -1)
            else:
                spp = torch.cat((spp, x1.view(num, -1)), 1)
        return spp
