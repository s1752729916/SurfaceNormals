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
augs_train = iaa.Sequential([
    # Geometric Augs
    iaa.Resize({"height": imgHeight, "width": imgWidth }, interpolation='nearest'),  # Resize image
    # iaa.Fliplr(0.5),
    # iaa.Flipud(0.5),
    # iaa.Rot90((0, 4)),
    # Blur and Noise
    # iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 1.5), name="gaus-blur")),
    # iaa.Sometimes(0.1, iaa.Grayscale(alpha=(0.0, 1.0), from_colorspace="RGB", name="grayscale")),
    # iaa.Sometimes(0.2, iaa.AdditiveLaplaceNoise(scale=(0, 0.1*255), per_channel=True, name="gaus-noise")),

    # Color, Contrast, etc.
    # iaa.Sometimes(0.2, iaa.Multiply((0.75, 1.25), per_channel=0.1, name="brightness")),
    # iaa.Sometimes(0.2, iaa.GammaContrast((0.7, 1.3), per_channel=0.1, name="contrast")),
    # iaa.Sometimes(0.2, iaa.AddToHueAndSaturation((-20, 20), name="hue-sat")),
    # iaa.Sometimes(0.3, iaa.Add((-20, 20), per_channel=0.5, name="color-jitter")),
])
input_only = [
    "simplex-blend", "add", "mul", "hue", "sat", "norm", "gray", "motion-blur", "gaus-blur", "add-element",
    "mul-element", "guas-noise", "lap-noise", "dropout", "cdropout"
]
dataset_tiny_white_cup_11_28 = dataloader_real.RealSurfaceNormalsDataset(input_I_sum_dir='/media/zjw/data/smq/samples/End2End/Tiny-White-Cup-11-28/I-sum',
                                                                         input_normal_dir='/media/zjw/data/smq/samples/End2End/Tiny-White-Cup-11-28/synthesis-normals',
                                                                         mask_dir= '/media/zjw/data/smq/samples/End2End/Tiny-White-Cup-11-28/masks',
                                                                         label_dir= '/media/zjw/data/smq/samples/End2End/Tiny-White-Cup-11-28/normals-png',transform=augs_train)
dataset_tiny_white_cup_11_30 = dataloader_real.RealSurfaceNormalsDataset(input_I_sum_dir='/media/zjw/data/smq/samples/End2End/Tiny-White-Cup-11-30/I-sum',
                                                                         input_normal_dir='/media/zjw/data/smq/samples/End2End/Tiny-White-Cup-11-30/synthesis-normals',
                                                                         mask_dir= '/media/zjw/data/smq/samples/End2End/Tiny-White-Cup-11-30/masks',
                                                                         label_dir= '/media/zjw/data/smq/samples/End2End/Tiny-White-Cup-11-30/normals-png',transform=augs_train)
dataset_plastic_cup_11_28 = dataloader_real.RealSurfaceNormalsDataset(input_I_sum_dir='/media/zjw/data/smq/samples/End2End/Plastic-Cup-11-28/I-sum',
                                                                         input_normal_dir='/media/zjw/data/smq/samples/End2End/Plastic-Cup-11-28/synthesis-normals',
                                                                         mask_dir='/media/zjw/data/smq/samples/End2End/Plastic-Cup-11-28/masks',
                                                                         label_dir= '/media/zjw/data/smq/samples/End2End/Plastic-Cup-11-28/normals-png',transform=augs_train)
dataset_plastic_cup_11_30 = dataloader_real.RealSurfaceNormalsDataset(input_I_sum_dir='/media/zjw/data/smq/samples/End2End/Plastic-Cup-11-30/I-sum',
                                                                         input_normal_dir='/media/zjw/data/smq/samples/End2End/Plastic-Cup-11-30/synthesis-normals',
                                                                         mask_dir='/media/zjw/data/smq/samples/End2End/Plastic-Cup-11-30/masks',
                                                                         label_dir= '/media/zjw/data/smq/samples/End2End/Plastic-Cup-11-30/normals-png',transform=augs_train)
dataset_middle_round_cup_11_28 = dataloader_real.RealSurfaceNormalsDataset(input_I_sum_dir='/media/zjw/data/smq/samples/End2End/Middle-Round-Cup-11-28/I-sum',
                                                                         input_normal_dir='/media/zjw/data/smq/samples/End2End/Middle-Round-Cup-11-28/synthesis-normals',
                                                                         mask_dir='/media/zjw/data/smq/samples/End2End/Middle-Round-Cup-11-28/masks',
                                                                         label_dir= '/media/zjw/data/smq/samples/End2End/Middle-Round-Cup-11-28/normals-png',transform=augs_train)
dataset_middle_round_cup_11_30 = dataloader_real.RealSurfaceNormalsDataset(input_I_sum_dir='/media/zjw/data/smq/samples/End2End/Middle-Round-Cup-11-30/I-sum',
                                                                         input_normal_dir='/media/zjw/data/smq/samples/End2End/Middle-Round-Cup-11-30/synthesis-normals',
                                                                         mask_dir = '/media/zjw/data/smq/samples/End2End/Middle-Round-Cup-11-30/masks',
                                                                         label_dir= '/media/zjw/data/smq/samples/End2End/Middle-Round-Cup-11-30/normals-png',transform=augs_train)
dataset_little_square_cup_11_28 = dataloader_real.RealSurfaceNormalsDataset(input_I_sum_dir='/media/zjw/data/smq/samples/End2End/Little-Square-Cup-11-28/I-sum',
                                                                         input_normal_dir='/media/zjw/data/smq/samples/End2End/Little-Square-Cup-11-28/synthesis-normals',
                                                                         mask_dir= '/media/zjw/data/smq/samples/End2End/Little-Square-Cup-11-28/masks',
                                                                         label_dir= '/media/zjw/data/smq/samples/End2End/Little-Square-Cup-11-28/normals-png',transform=augs_train)
dataset_little_square_cup_11_30 = dataloader_real.RealSurfaceNormalsDataset(input_I_sum_dir='/media/zjw/data/smq/samples/End2End/Little-Square-Cup-11-30/I-sum',
                                                                         input_normal_dir='/media/zjw/data/smq/samples/End2End/Little-Square-Cup-11-30/synthesis-normals',
                                                                         mask_dir= '/media/zjw/data/smq/samples/End2End/Little-Square-Cup-11-30/masks',
                                                                         label_dir= '/media/zjw/data/smq/samples/End2End/Little-Square-Cup-11-30/normals-png',transform=augs_train)
db_list = [dataset_tiny_white_cup_11_28,dataset_tiny_white_cup_11_30,dataset_plastic_cup_11_28,dataset_plastic_cup_11_30,dataset_middle_round_cup_11_28,dataset_middle_round_cup_11_30,dataset_little_square_cup_11_28,dataset_little_square_cup_11_30]
dataset = torch.utils.data.ConcatDataset(db_list)
#-- 3、create dataloader


testLoader = DataLoader(dataset,
                         num_workers=num_workers,
                         batch_size=batch_size,
                         drop_last=True,
                         pin_memory=pin_memory)
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
    theta_a_label = 180/np.pi*np.arctan2(label_t_rgb[:,:,1],label_t_rgb[:,:,0]) # 计算标签的方位角
    theta_z_label = 180/np.pi*np.arccos(label_t_rgb[:,:,2]) # 计算标签的天顶角
    # label_t_rgb[mask_t.squeeze(0)==0,:] = -1
    label_t_rgb = API.utils.normal_to_rgb(label_t_rgb)
    ax0.imshow(label_t_rgb)
    ax1 = plt.subplot(212)
    predict_norm = normal_vectors_norm.numpy().squeeze(0).transpose(1,2,0)
    predict_norm[mask_t.squeeze(0)==0,:] = -1
    theta_a_predict = 180/np.pi*np.arctan2(predict_norm[:,:,1],predict_norm[:,:,0]) # 计算预测的方位角
    theta_z_predict = 180/np.pi*np.arccos(predict_norm[:,:,2]) # 计算预测的天顶角

    # 计算方位角和天顶角误差
    theta_a_error = np.abs(theta_a_label-theta_a_predict)
    theta_z_error = np.abs(theta_z_label-theta_z_predict)
    theta_a_error[mask_t.squeeze(0)==0] = 0
    theta_z_error[mask_t.squeeze(0)==0] = 0
    mask_total_valid_pixels = torch.sum(mask_t.squeeze(0))/255
    theta_a_mean = np.sum(theta_a_error)/mask_total_valid_pixels
    theta_z_mean = np.sum(theta_z_error)/mask_total_valid_pixels


    predict_norm_rgb = API.utils.normal_to_rgb(predict_norm)
    predict_norm_rgb_tensor = torch.from_numpy(predict_norm_rgb.transpose(2,0,1))


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
    print(str(iter_num)+ "  theta_a_mean", theta_a_mean)
    print(str(iter_num)+ "  theta_z_mean", theta_z_mean)
    print(str(iter_num)+ "  loss_deg_mean", loss_deg_mean)
    print(str(iter_num)+ " loss_deg_median:", loss_deg_median)
    print(str(iter_num)+ " percentage_1:", percentage_1)
    print(str(iter_num)+ " percentage_2:", percentage_2)
    print(str(iter_num)+ " percentage_3:", percentage_3)



num_samples = len(testLoader)
print("test running loss:",running_loss/num_samples)
print("test running mean:",running_mean/num_samples)
print("test running median:",running_median/num_samples)



