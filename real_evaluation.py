'''
This file is used to test the performance of trained model on the real data set
'''
import os
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

###################### DataLoader #############################
#-- 1、 config parameters
imgHeight = 512
imgWidth = 512
batch_size = 1
num_workers = 4
shuffle_dataset = True
pin_memory = False
prefetch_factor = 1

#-- 2、create dataset
dataset_middle_round_cup = dataloader_real.RealSurfaceNormalsDataset(input_I_sum_dir='/media/smq/samples/Middle-Round-Cup-2/PolarImg-I-sum/8-Bit',
                                                                         input_normal_dir='/media/smq/samples/Middle-Round-Cup-2/synthesis-normals',
                                                                         label_dir= '/media/smq/samples/Middle-Round-Cup-2/Normals-PNG',
                                                                         mask_dir= '/media/smq/samples/Middle-Round-Cup-2/Masks')
#-- 3、create dataloader
testLoader = DataLoader(dataset_middle_round_cup,
                         num_workers=num_workers,
                         batch_size=batch_size,
                         drop_last=False,
                         pin_memory=pin_memory,
                         prefetch_factor=prefetch_factor
                         )

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



###################### Validation Cycle #############################

print('\nValidation:')
print('=' * 10)
criterion = loss_functions.loss_fn_cosine

model.eval() # eval mode, freeze params
running_loss = 0.0
running_mean = 0
running_median = 0
for iter_num, sample_batched in enumerate(tqdm(testLoader)):
    print('')
    inputs_t, label_t,mask_t = sample_batched
    inputs_t = inputs_t.to(device)
    label_t = label_t.to(device)

    with torch.no_grad():
        normal_vectors = model(inputs_t)
    label_t_numpy = label_t.numpy().squeeze(0).transpose(1,2,0)
    normal_vectors_norm = nn.functional.normalize(normal_vectors.double(), p=2, dim=1)
    loss = criterion(normal_vectors_norm, label_t.double(),reduction='sum')
    loss /= batch_size
    # calcute metrics
    inputs_t = inputs_t.detach().cpu()
    label_t = label_t.detach().cpu()
    normal_vectors_norm = normal_vectors_norm.detach().cpu()
    mask_t = mask_t.squeeze(1)  # To shape (batchSize, Height, Width)

    loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3 = loss_functions.metric_calculator_batch(
        normal_vectors_norm, label_t.double())

    running_mean += loss_deg_mean.item()
    running_median += loss_deg_median.item()
    running_loss += loss.item()


    fig = plt.figure()
    ax0 = plt.subplot(211)

    label_t_rgb = label_t.numpy().squeeze(0).transpose(1, 2, 0)
    # label_t_rgb[mask_t.squeeze(0)==0,:] = -1
    label_t_rgb = API.utils.normal_to_rgb(label_t_rgb)
    ax0.imshow(label_t_rgb)
    ax1 = plt.subplot(212)
    predict_norm = normal_vectors_norm.numpy().squeeze(0).transpose(1,2,0)

    predict_norm[mask_t.squeeze(0)==0,:] = -1


    predict_norm_rgb = API.utils.normal_to_rgb(predict_norm)

    predict_norm_rgb_tensor = torch.from_numpy(predict_norm_rgb.transpose(2,0,1))
    predict_norm_rgb_tensor = predict_norm_rgb_tensor.unsqueeze(0)
    label_t_rgb_tensor = torch.from_numpy(label_t_rgb.transpose(2,0,1))
    label_t_rgb_tensor = label_t_rgb_tensor.unsqueeze(0)

    # loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3 = loss_functions.metric_calculator_batch(
    #     predict_norm_rgb_tensor.double(), label_t_rgb_tensor.double())
    #
    #
    # running_mean += loss_deg_mean.item()
    # running_median += loss_deg_median.item()
    # running_loss += loss.item()
    ax1.imshow(predict_norm_rgb)
    # plt.show()
    API.utils.png_saver(os.path.join('/home/zjw/smq/SurfaceNormals/results',str(iter_num)+'-label.png'),label_t_rgb)
    API.utils.png_saver(os.path.join('/home/zjw/smq/SurfaceNormals/results',str(iter_num)+'-predict.png'),predict_norm_rgb)

    print(str(iter_num)+ "  loss_deg_mean", loss_deg_mean)
    print(str(iter_num)+ " loss_deg_median:", loss_deg_median)
    print(str(iter_num)+ " percentage_1:", percentage_1)
    print(str(iter_num)+ " percentage_2:", percentage_2)
    print(str(iter_num)+ " percentage_3:", percentage_3)



num_samples = len(testLoader)
print("test running loss:",running_loss/num_samples)
print("test running mean:",running_mean/num_samples)
print("test running median:",running_median/num_samples)



