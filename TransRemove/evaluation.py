# -*-coding:utf-8-*-

import os
import torch
import torch.nn as nn
from tqdm import tqdm
import API.utils
import numpy as np
from TransRemove.loss.loss import metric_calculator_batch

def evaluation(model,testLoader,device,criterion,epoch,resultPath = None,name=None,writer = None):
    ###################### Validation Cycle #############################


    model.eval() # eval mode, freeze params
    running_loss = 0.0
    running_mean_dolp = 0
    running_mean_aolp = 0
    for iter_num, sample_batched in enumerate(tqdm(testLoader)):
        # print('')
        params_t,label_params_t, label_t,mask_t = sample_batched
        params_t = params_t.to(device)
        label_params_t = label_params_t.to(device)
        label_t = label_t.to(device)

        with torch.no_grad():
            output_params = model(params_t)

        loss = criterion(output_params.double(), label_params_t.double(),mask_tensor = mask_t,reduction='mean',device = device)

        # calcute metrics

        mean_dolp,mean_aolp = metric_calculator_batch(output_params.double(), label_params_t.double(),mask_tensor=mask_t)
        running_loss += loss.item()
        running_mean_dolp += mean_dolp
        running_mean_aolp += mean_aolp


        # save validation pictures
        label_params_t_rgb = label_params_t.detach().cpu().numpy().squeeze(0)
        label_dolp_rgb = (label_params_t_rgb[0,:,:]*255).astype(np.uint8)
        label_aolp_rgb = (label_params_t_rgb[1,:,:]*255).astype(np.uint8)

        predict_params = output_params.detach().cpu().numpy().squeeze(0)
        mask_t = mask_t.squeeze(1)
        predict_params[:,mask_t.squeeze(0) == 0] = 0
        predict_dolp_rgb = (predict_params[0,:,:]*255).astype(np.uint8)
        predict_aolp_rgb = (predict_params[1,:,:]*255).astype(np.uint8)
        # save files for debug
        API.utils.png_saver(os.path.join(resultPath,str(iter_num).zfill(3) + '-dolp-label.png'),label_dolp_rgb)
        API.utils.png_saver(os.path.join(resultPath,str(iter_num).zfill(3) + '-aolp-label.png'),label_aolp_rgb)
        API.utils.png_saver(os.path.join(resultPath,str(iter_num).zfill(3) + '-dolp-predict.png'),predict_dolp_rgb)
        API.utils.png_saver(os.path.join(resultPath,str(iter_num).zfill(3) + '-aolp-predict.png'),predict_aolp_rgb)

    assert testLoader.batch_size == 1, 'testLoader batch size is need to be 1 instead of : "%d"' % (testLoader.batch_size)

    numsamples = len(testLoader)
    running_loss = running_loss/numsamples
    running_mean_dolp = running_mean_dolp/numsamples
    running_mean_aolp = running_mean_aolp/numsamples

    if(writer is not None):
        writer.add_scalar(name+'/'+'running_loss',running_loss,epoch)
        writer.add_scalar(name+'/'+'running_mean_dolp',running_mean_dolp,epoch)
        writer.add_scalar(name+'/'+'running_mean_aolp',running_mean_aolp,epoch)
    return running_loss,running_mean_dolp,running_mean_aolp
