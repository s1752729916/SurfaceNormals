'''
This file is used to test single picture without dataloader
'''



# -*-coding:utf-8-*-
from dataloader import dataloaderDAN
from imgaug import augmenters as iaa
import API.utils
from TransMVS.model import smqFusion
import os
from termcolor import colored
import torch
import torch.nn as nn
from imgaug import augmenters as iaa
from torch.utils.data import DataLoader
from tqdm import tqdm
import loss_functions
import API.utils
import torch.nn.functional as F
import numpy as np


def evaluation(model, testLoader, device, criterion, epoch, resultPath=None, name=None, writer=None, use_atten=False):
    ###################### Validation Cycle #############################

    model.eval()  # eval mode, freeze params
    running_loss = 0.0
    running_mean = 0
    running_median = 0
    running_percentage_1 = 0
    running_percentage_2 = 0
    running_percentage_3 = 0
    mean_list = []
    for iter_num, sample_batched in enumerate(tqdm(testLoader)):
        # print('')
        params_t, normals_t, label_t, mask_t = sample_batched
        params_t = params_t.to(device)
        normals_t = normals_t.to(device)

        label_t = label_t.to(device)

        with torch.no_grad():
            normal_vectors, atten_map = model(params_t, normals_t)

        normal_vectors_norm = nn.functional.normalize(normal_vectors.double(), p=2, dim=1)
        normal_vectors_norm = normal_vectors_norm
        # loss = criterion(normal_vectors_norm, label_t.double(),reduction='sum',device=device)
        loss = criterion(normal_vectors_norm, label_t.double(), mask_tensor=mask_t, atten_map=atten_map,aolp=None,
                         reduction='sum', device=device, use_atten=use_atten)

        # calcute metrics
        label_t = label_t.detach().cpu()
        normal_vectors_norm = normal_vectors_norm.detach().cpu()

        loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3 = loss_functions.metric_calculator_batch(
            normal_vectors_norm, label_t.double())
        running_mean += loss_deg_mean.item()
        running_median += loss_deg_median.item()
        mean_list.append(loss_deg_mean.item())
        running_loss += loss.item()
        running_percentage_1 += percentage_1.item()
        running_percentage_2 += percentage_2.item()
        running_percentage_3 += percentage_3.item()

        # save validation pictures
        label_t = F.interpolate(label_t, size=(1028, 1232), mode='bilinear', align_corners=False)
        label_t_rgb = label_t.numpy().squeeze(0).transpose(1, 2, 0)
        label_t_rgb = API.utils.normal_to_rgb(label_t_rgb)
        predict_norm = normal_vectors_norm.numpy().squeeze(0).transpose(1, 2, 0)
        mask_t = mask_t.squeeze(1)
        predict_norm[mask_t.squeeze(0) == 0, :] = -1
        predict_norm = predict_norm.transpose(2,0,1)
        predict_norm = torch.from_numpy(predict_norm).unsqueeze(0)
        predict_norm = F.interpolate(predict_norm, size=(1028, 1232), mode='bilinear', align_corners=False)
        predict_norm = predict_norm.numpy().squeeze(0).transpose(1, 2, 0)


        predict_norm_rgb = API.utils.normal_to_rgb(predict_norm)
        atten_map_rgb = atten_map.detach().cpu().numpy().squeeze(0).transpose(1, 2, 0)
        atten_map_rgb = atten_map_rgb * 255
        atten_map_rgb = atten_map_rgb.astype(np.uint8)
        # API.utils.png_saver(os.path.join(resultPath, str(iter_num).zfill(3) + '-label.png'),
        #                     label_t_rgb)
        API.utils.png_saver(os.path.join(resultPath, str(iter_num).zfill(3) + '-predict.png'),
                            predict_norm_rgb)
        # API.utils.png_saver(os.path.join(resultPath, str(iter_num).zfill(3) + '-atten.png'),
        #                     atten_map_rgb)

    assert testLoader.batch_size == 1, 'testLoader batch size is need to be 1 instead of : "%d"' % (
        testLoader.batch_size)

    numsamples = len(testLoader)
    running_loss = running_loss / numsamples
    running_mean = running_mean / numsamples
    running_median = running_median / numsamples
    running_percentage_1 = running_percentage_1 / numsamples
    running_percentage_2 = running_percentage_2 / numsamples
    running_percentage_3 = running_percentage_3 / numsamples
    if (writer is not None):
        writer.add_scalar(name + '/' + 'running_mean', running_mean, epoch)
        writer.add_scalar(name + '/' + 'running_median', running_median, epoch)
        writer.add_scalar(name + '/' + 'running_loss', running_loss, epoch)
        writer.add_scalar(name + '/' + 'running_percentage_1', running_percentage_1, epoch)
        writer.add_scalar(name + '/' + 'running_percentage_2', running_percentage_2, epoch)
        writer.add_scalar(name + '/' + 'running_percentage_3', running_percentage_3, epoch)
    print('mean list:', mean_list)
    return running_loss, running_mean, running_median, running_percentage_1, running_percentage_2, running_percentage_3


# set parameters
imgHeight = 512
imgWidth = 512
batch_size = 6
num_workers = 2
validation_split = 0.1
shuffle_dataset = True
pin_memory = False
prefetch_factor = 8

# create dataset
augs_train = iaa.Sequential([
    # Geometric Augs
    iaa.Resize({"height": imgHeight, "width": imgWidth }, interpolation='nearest'),  # Resize image
])
input_only = [
    "simplex-blend", "add", "mul", "hue", "sat", "norm", "gray", "motion-blur", "gaus-blur", "add-element",
    "mul-element", "guas-noise", "lap-noise", "dropout", "cdropout"
]
root_dir = '/media/disk2/smq_data/samples/PolarMVS/7-24/cat-big'
dataset = dataloaderDAN.DANSurfaceDataset(dolp_dir = os.path.join(os.path.join(root_dir,'params'),'DoLP'),
                                                                                  aolp_dir = os.path.join(os.path.join(root_dir,'params'),'AoLP'),
                                                                                  synthesis_normals_dir=os.path.join(root_dir,'synthesis-normals'),
                                                                                  mask_dir= os.path.join(root_dir,'masks'),
                                                                                  label_dir= os.path.join(root_dir,'normals-png'), transform=augs_train)
testLoader = DataLoader(dataset,
                         batch_size=1,
                         num_workers=num_workers,
                         drop_last=True,
                         pin_memory=pin_memory)
###################### ModelLoad #############################
#-- 1、 config parameters
backbone_model = 'resnet50'
device = torch.device("cpu")
sync_bn = False  # this is for Multi-GPU synchronize batch normalization
numClasses = 3

#-- 2、create model
model = smqFusion(backbone = backbone_model,num_classes=3,device=device,sync_bn=False)


#-- 3、load model params
CHECKPOINT_DIR = '/media/disk2/smq_data/samples/TransMVS'
checkpoint_path = os.path.join(CHECKPOINT_DIR,'check-point-epoch-0020.pth')

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

criterion = loss_functions.my_loss_cosine

# evaluation
model.eval() # eval mode, freeze params


running_loss,running_mean,running_median,running_percentage_1,running_percentage_2,running_percentage_3 = evaluation(model = model,
        testLoader= testLoader,device=device,criterion=criterion,use_atten=False,epoch = 0,name = 'pot',writer=None,resultPath='/home/robotlab/smq/SurfaceNormals/results/bear-2')

print(running_mean)