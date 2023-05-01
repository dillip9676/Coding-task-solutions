# -*- coding: utf-8 -*-
"""task3_v4.ipynb


Original file is located at
    https://colab.research.google.com/drive/1wFSRN_y-6NcTgQJRSJJhXErtUuWpAxHO

## Task Description

'''
develop a model based on the onnx file in model/model.onnx 

Note:
    - initialize the convolutions layer with uniform xavier           
    - initialize the linear layer with a normal distribution (mean=0.0, std=1.0)  
    - initialize all biases with zeros                                           
    - use batch norm wherever is relevant
    - use random seed 8                                                          
    - use default values for anything unspecified                          
'''
"""

from google.colab import drive
drive.mount('/content/gdrive')

import os
os.chdir('/content/gdrive/My Drive/Artificient/cv_applicant_tasks')
os.getcwd()

!pip install torchinfo
!pip install onnx
!pip install onnx onnx2pytorch

import numpy as np
import torch

import onnx
from onnx2pytorch import ConvertModel
from torchinfo import summary

torch.manual_seed(8)    # DO NOT MODIFY!
np.random.seed(8)   # DO NOT MODIFY!

model_path = "model/model.onnx"
model = onnx.load(model_path)

# Convert ONNX model to PyTorch model
pytorch_model = ConvertModel(model)

# empty list to store conv2D layers and used in sequential layer
modules_to_add = []

# Initializing weights of convolution layer with Xavier uniform
for name, module in pytorch_model.named_modules():

    if isinstance(module, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(module.weight)
        
        modules_to_add.append((name, module))

# replace conv2d layer with sequential layer (conv2d and BatchNorm2d layers) 
for name, module in modules_to_add:
  
    bn = torch.nn.BatchNorm2d(module.out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    conv_bn_module = torch.nn.Sequential(module, bn)                         
    pytorch_model.add_module(name, conv_bn_module)

# Initialize weights of linear layers with normal distribution
for module in pytorch_model.modules():
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

# Initialize biases with zeros
for module in pytorch_model.modules():
    if hasattr(module, 'bias') and module.bias is not None:
        torch.nn.init.zeros_(module.bias)

print(pytorch_model)

# view the architecture of the pytorch_model
summary(pytorch_model, input_size=(1, 3, 160, 320))

