# -*- coding: utf-8 -*-
"""task1_v3.ipynb


Original file is located at
    https://colab.research.google.com/drive/1fi9aZGqPL6RN1Glb-AUIM-XKJHkfn9XC

## Task description
Write a code using pytorch to replicate a grouped 2D convolution layer based on the original 2D convolution. 

The common way of using grouped 2D convolution layer in Pytorch is to use 
torch.nn.Conv2d(groups=n), where n is the number of groups.

However, it is possible to use a stack of n torch.nn.Conv2d(groups=1) to replicate the same
result. The wights must be copied and be split between the convs in the stack.

You can use:
    - use default values for anything unspecified  
    - all available functions in NumPy and Pytorch
    - the custom layer must be able to take all parameters of the original nn.Conv2d
"""

from google.colab import drive
drive.mount('/content/gdrive')

import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(8)    # DO NOT MODIFY!
np.random.seed(8)   # DO NOT MODIFY!

x = torch.randn(2, 64, 100, 100)

# original 2d grouped convolution
grouped_layer = nn.Conv2d(64, 128, 3, stride=1, padding=1, groups=16, bias=True)

w_torch = grouped_layer.weight
b_torch = grouped_layer.bias

y = grouped_layer(x)

# custom layer to replicate a grouped 2D convolution layer
class CustomGroupedConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(CustomGroupedConv2D, self).__init__()
        
        # parameters
        self.groups = groups
        self.convs = nn.ModuleList()    # empty list to hold "nn.con2d" layers
        self.in_channels = in_channels
        self.out_channels = out_channels

        # splitting input and output channels 
        in_groups = self.in_channels // self.groups
        out_groups = self.out_channels // self.groups 

        # creating a stack of conv2d layers equal no of groups
        for i in range(groups):
            self.convs.append(nn.Conv2d(in_groups, out_groups, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=1, bias=bias))
        

    
    def forward(self, x):
        splits = torch.split(x, x.size(1) // self.groups, dim=1)                         # splitting the input tensor into groups along the channel axis
        out = torch.cat([conv(split) for split, conv in zip(splits, self.convs)], dim=1) # concatenate all the conv2d layers to replicate grouped conv2d layer
        return out

# custom grouped conv2d
custom_grouped_layer = CustomGroupedConv2D(64, 128, 3, stride=1, padding=1, groups=16, bias=True)
groups = custom_grouped_layer.groups

# copying weights and bias from the original 2d convolutin
for i in range(groups):
    custom_grouped_layer.convs[i].weight.data = grouped_layer.weight.data[i*(128//16):(i+1)*(128//16), :, :, :]
    custom_grouped_layer.convs[i].bias.data = grouped_layer.bias.data[i*(128//16):(i+1)*(128//16)]

y_custom = custom_grouped_layer(x)

# To check weather the outputs are equal
print(torch.allclose(y, y_custom, rtol=1e-6, atol=1e-6))

