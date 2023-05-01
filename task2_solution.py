# -*- coding: utf-8 -*-
"""task2_v2.ipynb

Original file is located at
    https://colab.research.google.com/drive/14HYC_J_S3OTRuq2g7TaNEZXkwTB4-SuN


## Task description

Use the following augmentation methods on the sample image under data/sample.png
and save the result under this path: 'data/sample_augmented.png'

Note:
    - use torchvision.transforms
    - use the following augmentation methods with the same order as below:
        * affine: degrees: ±5, 
                  translation= 0.1 of width and height, 
                  scale: 0.9-1.1 of the original size
        * rotation ±5 degrees,
        * horizontal flip with a probablity of 0.5
        * center crop with height=320 and width=640
        * resize to height=160 and width=320
        * color jitter with:  brightness=0.5, 
                              contrast=0.5, 
                              saturation=0.4, 
                              hue=0.2
    - use default values for anything unspecified
"""

from google.colab import drive
drive.mount('/content/gdrive')

import torch
import torchvision
from torchvision import transforms as T
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

torch.manual_seed(8)
np.random.seed(8)

import os
os.chdir('C:\Users\dilli\Desktop\Artificient\cv_applicant_tasks\data')
os.getcwd()

img = cv2.imread('data/sample.png',)

plt.imshow(img[...,::-1])

# Convert to PIL Image because torchvision.transforms expects the input to be a PIL Image object or tesnors
img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# squence of tranformations on image
transform = T.Compose([
    T.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),

    T.RandomRotation(degrees=5),     

    T.RandomHorizontalFlip(p=0.5),   

    T.CenterCrop(size=(320, 640)),   

    T.Resize(size=(160, 320)),       

    T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.2) 
])

augmented_image = transform(img)
augmented_image = np.array(augmented_image)

plt.imshow(augmented_image)

# Save the augmented image
cv2.imwrite('data/sample_augmented4.png', augmented_image)


"""we can combine multiple transformations into a single pipeline using **Compose** and **sequentital** but both are slightly different in their application.

**torchvision.transforms.Compose** is a functional composition that takes a list of transformations and applies them sequentially to an image. It is useful for applying a fixed set of transformations to a dataset.

**torch.nn.Sequential** is a class that creates a container for a sequence of operations that can be applied to an input tensor, where each operation is a callable module. It is useful for defining complex models with multiple layers.

For the above task i considered **Compose** method because all sequence of operations can be saved and loaded as single module
"""