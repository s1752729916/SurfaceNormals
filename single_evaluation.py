'''
This file is used to test single picture without dataloader
'''
import os

import imageio
import matplotlib.pyplot as plt
from termcolor import colored
import numpy as np
import torch
import torch.nn as nn
from imgaug import augmenters as iaa
from torch.utils.data import DataLoader,SubsetRandomSampler
from torchvision import transforms
from tqdm import tqdm

import API.utils
from modeling import deeplab
import dataloader_real
import loss_functions


# define paths
I_sum_path = '/home/zjw/smq/SurfaceNormals/test/12_8_MONO_8bit_Angle_001_Sum.tif'
normal_0_path = '/home/zjw/smq/SurfaceNormals/test/normal_0.png'
normal_1_path = '/home/zjw/smq/SurfaceNormals/test/normal_1.png'
normal_2_path = '/home/zjw/smq/SurfaceNormals/test/normal_2.png'
normal_3_path = '/home/zjw/smq/SurfaceNormals/test/normal_3.png'

# load imgs
I_sum = imageio.imread(I_sum_path)
normal_0 = imageio.imread(normal_0_path)
normal_1 = imageio.imread(normal_1_path)
normal_2 = imageio.imread(normal_2_path)
normal_3 = imageio.imread(normal_3_path)

# preprocessing
height = I_sum.shape[0]
width = I_sum.shape[1]
input_img_arr = np.zeros([13, height, width], dtype=np.uint8)  # shape is (13 x H x W)
input_img_arr[0, :, :] = I_sum
input_img_arr[1:4, :, :] = normal_0.transpose(2, 0, 1)  # 3 x H x W
input_img_arr[4:7, :, :] = normal_1.transpose(2, 0, 1)
input_img_arr[7:10, :, :] = normal_2.transpose(2, 0, 1)
input_img_arr[10:13, :, :] = normal_3.transpose(2, 0, 1)
input_tensor = transforms.ToTensor()(input_img_arr.copy().transpose(1, 2, 0))  # ToTensor contains the normalization process
input_tensor = input_tensor.unsqueeze(0)
###################### ModelLoad #############################
#-- 1、 config parameters
backbone_model = 'resnet50'
device = torch.device("cpu")
sync_bn = False  # this is for Multi-GPU synchronize batch normalization
numClasses = 3

#-- 2、create model
model = deeplab.DeepLab(num_classes=numClasses,
                        backbone=backbone_model,
                        sync_bn=sync_bn,
                        freeze_bn=False)

#-- 3、load model params
CHECKPOINT_DIR = '/home/zjw/smq/SurfaceNormals/CheckPoints'
checkpoint_path = os.path.join(CHECKPOINT_DIR,'check-point-epoch-0007.pth')

if not os.path.isfile(checkpoint_path):
    raise ValueError('Invalid path to the given weights file for transfer learning.\
            The file {} does not exist'.format(checkpoint_path))
CHECKPOINT = torch.load(checkpoint_path,'cpu')
if 'model_state_dict' in CHECKPOINT:
    # Our weights file with various dicts
    model.load_state_dict(CHECKPOINT['model_state_dict'])
elif 'state_dict' in CHECKPOINT:
    # Original Author's checkpoint
    CHECKPOINT['state_dict'].pop('decoder.last_conv.8.weight')
    CHECKPOINT['state_dict'].pop('decoder.last_conv.8.bias')
    model.load_state_dict(CHECKPOINT['state_dict'], strict=False)
else:
    # Our old checkpoint containing only model's state_dict()
    model.load_state_dict(CHECKPOINT)


# evaluation
model.eval() # eval mode, freeze params

with torch.no_grad():
    output_t = model(input_tensor)
normal_vectors_norm = nn.functional.normalize(output_t.double(), p=2, dim=1)
predict_norm = normal_vectors_norm.numpy().squeeze(0).transpose(1,2,0)
predict_norm_rgb = API.utils.normal_to_rgb(predict_norm)
predict_norm_rgb_tensor = torch.from_numpy(predict_norm_rgb.transpose(2,0,1))
plt.figure()
plt.imshow(predict_norm_rgb)
plt.show()


