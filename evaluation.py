# -*-coding:utf-8-*-

import os
from termcolor import colored
import torch
import torch.nn as nn
from imgaug import augmenters as iaa
from torch.utils.data import DataLoader,SubsetRandomSampler
from tqdm import tqdm
from modeling import deeplab
from dataloader import dataloaderIDA,dataloaderI,dataloaderIDAN
import loss_functions
import API.utils
import numpy as np
import random

def evaluation(model,testLoader,device,criterion,resultPath = None):
    ###################### Validation Cycle #############################


    model.eval() # eval mode, freeze params
    running_loss = 0.0
    running_mean = 0
    running_median = 0
    running_percentage_1 = 0
    running_percentage_2 = 0
    running_percentage_3 = 0
    for iter_num, sample_batched in enumerate(tqdm(testLoader)):
        # print('')
        params_t,normals_t, label_t,mask_t = sample_batched
        params_t = params_t.to(device)
        normals_t = normals_t.to(device)

        label_t = label_t.to(device)

        with torch.no_grad():
            normal_vectors = model(params_t,normals_t)

        normal_vectors_norm = nn.functional.normalize(normal_vectors.double(), p=2, dim=1)
        # loss = criterion(normal_vectors_norm, label_t.double(),reduction='sum',device=device)
        loss = criterion(normal_vectors_norm, label_t.double(),reduction='sum')

        # calcute metrics
        label_t = label_t.detach().cpu()
        normal_vectors_norm = normal_vectors_norm.detach().cpu()

        loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3 = loss_functions.metric_calculator_batch(
            normal_vectors_norm, label_t.double())
        running_mean += loss_deg_mean.item()
        running_median += loss_deg_median.item()
        running_loss += loss.item()
        running_percentage_1 += percentage_1.item()
        running_percentage_2 +=percentage_2.item()
        running_percentage_3 +=percentage_3.item()

        # save validation pictures
        label_t_rgb = label_t.numpy().squeeze(0).transpose(1, 2, 0)
        label_t_rgb = API.utils.normal_to_rgb(label_t_rgb)
        predict_norm = normal_vectors_norm.numpy().squeeze(0).transpose(1, 2, 0)
        mask_t = mask_t.squeeze(1)
        predict_norm[mask_t.squeeze(0) == 0, :] = -1
        predict_norm_rgb = API.utils.normal_to_rgb(predict_norm)
        API.utils.png_saver(os.path.join(resultPath, str(iter_num).zfill(3) + '-label.png'),
                            label_t_rgb)
        API.utils.png_saver(os.path.join(resultPath, str(iter_num).zfill(3) + '-predict.png'),
                            predict_norm_rgb)
    assert testLoader.batch_size == 1, 'testLoader batch size is need to be 1 instead of : "%d"' % (testLoader.batch_size)

    numsamples = len(testLoader)
    running_loss = running_loss/numsamples
    running_mean = running_mean/numsamples
    running_median = running_median/numsamples
    running_percentage_1 = running_percentage_1/numsamples
    running_percentage_2 = running_percentage_2/numsamples
    running_percentage_3 = running_percentage_3/numsamples
    return running_loss,running_mean,running_median,running_percentage_1,running_percentage_2,running_percentage_3
