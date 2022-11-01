import os
from glob import glob
import cv2
import numpy as np
import torch
import yaml

config = None
with open('./convert.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


image_files = config['from']
print(image_files)

images = []
for image_file in image_files:
  img = torch.load(image_file).tolist()
  images += img


print('Total number of images: ', len(images))


exposure_path = config['to']

if not os.path.exists(exposure_path):
    os.makedirs(exposure_path)


import torch
import torchvision

images_tensor = torch.tensor(images)

for i in range(images_tensor.size(0)):
    torchvision.utils.save_image(images_tensor[i, :, :, :], f'{exposure_path}{i:04d}.png')