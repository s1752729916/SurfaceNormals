'''
This module contains the loss functions used to train the surface normals estimation models.
'''

import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from losses.TVLoss import TVLoss
from losses.GradientLoss import Gradient_Net



def loss_fn_mae(input_vec, target_vec,mask_tensor, reduction='mean',device = None):
    '''A loss function for use with surface normals estimation.
    Calculates the loss between 2 vectors. Both should be of the same size.
    Arguments:
        input_vec {tensor} -- The 1st vectors with whom loss is to be calculated
                              The dimensions of the matrices are expected to be (batchSize, 2, height, width).
        target_vec {tensor } -- The 2nd vectors with whom loss is to be calculated
                                The dimensions of the matrices are expected to be (batchSize, 2, height, width).
        mask_tensor {tensor } --  The mask tensor used to specify where losses are calculated
                                The dimensions of the matrices are expected to be (batchSize, height, width).
    Keyword Arguments:
        reduction {str} -- Can have values 'mean' and 'none'.
                           If 'mean' is passed, the mean of all elements is returned
                           if 'none' is passed, a matrix of all  losses is returned, same size as input.
                           (default: {'sum'})
    Raises:
        Exception -- Exception is an invalid reduction is passed
    Returns:
        tensor -- A single mean value of cosine loss or a matrix of elementwise cosine loss.
    '''

    bs,c,h,w = target_vec.shape
    mask_invalid_pixels = torch.all(mask_tensor<255,dim=1)
    total_valid_pixels = (~mask_invalid_pixels).sum()
    target_vec = target_vec.permute(1,0,2,3)
    target_vec[:,mask_invalid_pixels] = 0
    target_vec = target_vec.permute(1,0,2,3)
    input_vec = input_vec.permute(1,0,2,3)
    input_vec[:,mask_invalid_pixels] = 0
    input_vec = input_vec.permute(1,0,2,3)

    target_vec = target_vec.view(-1)
    input_vec = input_vec.view(-1)
    l1_loss = nn.L1Loss(reduction='sum')

    loss = l1_loss(input_vec,target_vec)

    if (reduction=='sum'):
        return loss
    elif (reduction=='mean'):
        return loss/total_valid_pixels
    else:
        raise NotImplementedError('this reduction is not implemented!')

def metric_calculator_batch(input_vec, target_vec, mask_tensor=None):
    """Calculate mean, median and angle error between prediction and ground truth
    Args:
        input_vec (tensor): The 1st vectors with whom cosine loss is to be calculated
                            The dimensions of are expected to be (batchSize, 3, height, width).
        target_vec (tensor): The 2nd vectors with whom cosine loss is to be calculated.
                             This should be GROUND TRUTH vector.
                             The dimensions are expected to be (batchSize, 3, height, width).
        mask (tensor): The pixels over which loss is to be calculated. Represents VALID pixels.
                             The dimensions are expected to be (batchSize, height, width).
    Returns:
        float: The mean error in 2 surface normals in degrees
        float: The median error in 2 surface normals in degrees
        float: The percentage of pixels with error less than 11.25 degrees
        float: The percentage of pixels with error less than 22.5 degrees
        float: The percentage of pixels with error less than 30 degrees
    """
    # new_input = torch.zeros(size=(input_vec.size()[0],3,input_vec.size()[2],input_vec.size()[3]))
    # new_input[:,0,:,:] = torch.cos(input_vec[:,1,:,:])*torch.sin(input_vec[:,0,:,:])
    # new_input[:,1,:,:] = torch.sin(input_vec[:,1,:,:])*torch.sin(input_vec[:,0,:,:])
    # new_input[:,2,:,:] = torch.cos(input_vec[:,0,:,:])
    # input_vec = new_input

    if len(input_vec.shape) != 4:
        raise ValueError('Shape of tensor must be [B, C, H, W]. Got shape: {}'.format(input_vec.shape))
    if len(target_vec.shape) != 4:
        raise ValueError('Shape of tensor must be [B, C, H, W]. Got shape: {}'.format(target_vec.shape))


    bs,c,h,w = target_vec.shape
    mask_invalid_pixels = torch.all(mask_tensor<255,dim=1)
    total_valid_pixels = (~mask_invalid_pixels).sum()
    target_vec = target_vec.permute(1,0,2,3)
    target_vec[:,mask_invalid_pixels] = 0
    target_vec = target_vec.permute(1,0,2,3)
    input_vec = input_vec.permute(1,0,2,3)
    input_vec[:,mask_invalid_pixels] = 0
    input_vec = input_vec.permute(1,0,2,3)

    target_dolp = target_vec[:,0,:,:]
    target_aolp = target_vec[:,1,:,:]
    input_dolp = input_vec[:,0,:,:]
    input_aolp = input_vec[:,1,:,:]

    target_dolp = target_dolp.view(bs,-1)
    target_aolp = target_aolp.view(bs,-1)
    input_dolp = input_dolp.view(bs,-1)
    input_aolp = input_aolp.view(bs,-1)
    l1_loss = nn.L1Loss(reduction='sum') # average value of all elements
    dolp_loss = l1_loss(input_dolp,target_dolp)/total_valid_pixels
    aolp_loss = l1_loss(input_aolp,target_aolp)/total_valid_pixels
    aolp_loss = aolp_loss*180
    return dolp_loss,aolp_loss

if __name__ == '__main__':
    input_vec = torch.ones(8,2,512,512)
    target_vec = torch.zeros(8,2,512,512)
    mask_vec = torch.ones(8,1,512,512)*255
    loss_dolp,loss_aolp = metric_calculator_batch(input_vec,target_vec,mask_vec)
    print(loss_dolp,loss_aolp)




