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
def my_loss_cosine(input_vec,target_vec,atten_map,aolp,mask_tensor = None,reduction = 'sum',smooth_item = False,device = None,use_atten = False):
    # 输入进来的input_vec是天顶角和方位角2个通道的数据，需要转换成正常的余弦损失函数计算所需要的法向量的形式
    # input_vec -- (batchSize,2,height,width):(sin_theta,cos_theta,sin_phi,cos_phi)
    # new_input = torch.zeros(size=(input_vec.size()[0],3,input_vec.size()[2],input_vec.size()[3])).to(device)
    # new_input[:,0,:,:] = torch.cos(input_vec[:,1,:,:])*torch.sin(input_vec[:,0,:,:])
    # new_input[:,1,:,:] = torch.sin(input_vec[:,1,:,:])*torch.sin(input_vec[:,0,:,:])
    # new_input[:,2,:,:] = torch.cos(input_vec[:,0,:,:])
    new_input = input_vec
    cosine_loss = loss_fn_cosine(new_input,target_vec,atten_map = atten_map,mask_tensor=mask_tensor,reduction=reduction,device=device,use_atten = use_atten)
    aolp_loss = loss_aolp(new_input,aolp,mask_tensor,atten_map)
    # print('cosine_loss:',cosine_loss)
    # print('aolp_loss:',aolp_loss)
    loss = cosine_loss + aolp_loss*0.05
    return loss

def loss_aolp(input_vec,aolp,mask_tensor,atten_map):
    atten_map = atten_map.squeeze(1)
    aolp = aolp.squeeze(1) * torch.pi
    aolp_0 = aolp + torch.pi / 2
    aolp_1 = aolp - torch.pi / 2
    aolp_0 = torch.remainder(aolp_0,torch.pi * 2)
    aolp_1 = torch.remainder(aolp_1,torch.pi * 2)

    mask_invalid_pixels = torch.all(mask_tensor < 255, dim=1)
    y = input_vec[:,1,:,:]
    x = input_vec[:,0,:,:]
    phi = torch.atan2(y,x) # (batchSize,H,W) (-pi,pi)
    phi = torch.remainder(phi,torch.pi * 2)
    # aolp[mask_invalid_pixels] = 0.0  #这句有错
    error_0 = torch.min(torch.abs(phi - aolp_0),
                              torch.pi * 2 - torch.abs(phi - aolp_0)) # (bs,H,W)
    error_1 = torch.min(torch.abs(phi - aolp_1),
                              torch.pi * 2 - torch.abs(phi - aolp_1))
    error = torch.min(error_0,error_1)
    error = error * (atten_map)
    error[mask_invalid_pixels] = 0.0
    loss = torch.sum(error)
    total_valid_pixels = (~mask_invalid_pixels).sum()
    # loss = loss / total_valid_pixels
    return loss


def loss_fn_cosine(input_vec, target_vec,mask_tensor, reduction='sum',device = None,atten_map = None,use_atten = False):
    '''A cosine loss function for use with surface normals estimation.
    Calculates the cosine loss between 2 vectors. Both should be of the same size.
    Arguments:
        input_vec {tensor} -- The 1st vectors with whom cosine loss is to be calculated
                              The dimensions of the matrices are expected to be (batchSize, 3, height, width).
        target_vec {tensor } -- The 2nd vectors with whom cosine loss is to be calculated
                                The dimensions of the matrices are expected to be (batchSize, 3, height, width).
        mask_tensor {tensor } --  The mask tensor used to specify where losses are calculated
                                The dimensions of the matrices are expected to be (batchSize, height, width).
    Keyword Arguments:
        reduction {str} -- Can have values 'elementwise_mean' and 'none'.
                           If 'elemtwise_mean' is passed, the mean of all elements is returned
                           if 'none' is passed, a matrix of all cosine losses is returned, same size as input.
                           (default: {'elementwise_mean'})
    Raises:
        Exception -- Exception is an invalid reduction is passed
    Returns:
        tensor -- A single mean value of cosine loss or a matrix of elementwise cosine loss.
    '''
    atten_map = atten_map.squeeze(1)
    mask_invalid_pixels = torch.all(mask_tensor<255,dim=1)

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    lamda_1 = 0
    lamda_2 = 3
    lamda_3 = 0
    tvLoss = TVLoss()
    if(use_atten ==True):
        loss_cos = (1.0 - cos(input_vec, target_vec))*(1 + atten_map*lamda_2)
    else:
        loss_cos = (1.0 - cos(input_vec, target_vec))*(1 + atten_map*lamda_2)
    loss_cos = (1.0-cos(input_vec,target_vec))
    # gredient item
    # if(use_atten ==True):
        # gradient_model = Gradient_Net(device).to(device)
        # g_target = gradient_model(target_vec)
        #
        #
        # g_output = gradient_model(input_vec)
        # loss_gradient = (g_target-g_output)*(g_target-g_output)*(1+atten_map*lamda_2)
        # # temp = input_vec.detach().cpu().numpy()
        # # temp = temp[0,:,:,:].transpose(1,2,0)
        # # plt.imshow(temp[:,:,0])
        # # plt.show()
        # loss_gradient[mask_invalid_pixels] = 0
        # loss_gradient_sum = loss_gradient.sum()

        # tv_loss = TVLoss(TVLoss_weight=1)
        # gradient_model = Gradient_Net(device).to(device)
        # g_output = gradient_model(input_vec)
        # loss_gradient = g_output*(1 + atten_map * lamda_2)



    tv_vector = (target_vec+1)*127.5
    tv_loss = tvLoss(tv_vector)




    # calculate loss only on valid pixels
    # mask_invalid_pixels = (target_vec[:, 0, :, :] == -1.0) & (target_vec[:, 1, :, :] == -1.0) & (target_vec[:, 2, :, :] == -1.0)
    # mask_invalid_pixels = torch.all(mask_tensor==0,dim=)
    loss_cos[mask_invalid_pixels] = 0.0
    loss_cos_sum = loss_cos.sum()
    total_valid_pixels = (~mask_invalid_pixels).sum()
    # print(total_valid_pixels)
    if(use_atten ==True):
        error_sum = (loss_cos_sum + 0 + tv_loss*lamda_3)
    else:
        error_sum = loss_cos_sum + tv_loss*lamda_3

    error_output = error_sum / total_valid_pixels
    if reduction == 'elementwise_mean':
        loss_cos = error_output
    elif reduction == 'sum':
        loss_cos = error_sum
    elif reduction == 'none':
        loss_cos = loss_cos
    else:
        raise Exception(
            'Invalid value for reduction  parameter passed. Please use \'elementwise_mean\' or \'none\''.format())
    return loss_cos


def metric_calculator_batch(input_vec, target_vec, mask=None):
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

    INVALID_PIXEL_VALUE = -1/np.sqrt(3)+1e-4  # All 3 channels should have this value

    mask_valid_pixels = ~(torch.all(target_vec < INVALID_PIXEL_VALUE, dim=1))


    total_valid_pixels = mask_valid_pixels.sum()
    # print("total_valid_pixels:",total_valid_pixels)
    if (total_valid_pixels == 0):
        print('[WARN]: Image found with ZERO valid pixels to calc metrics')
        return torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), mask_valid_pixels

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    loss_cos = cos(input_vec, target_vec)

    # Taking torch.acos() of 1 or -1 results in NaN. We avoid this by small value epsilon.
    eps = 1e-10
    loss_cos = torch.clamp(loss_cos, (-1.0 + eps), (1.0 - eps))
    loss_rad = torch.acos(loss_cos)
    loss_deg = loss_rad * (180.0 / math.pi)

    # Mask out all invalid pixels and calc mean, median
    loss_deg = loss_deg[mask_valid_pixels.bool()]
    temp = torch.min(loss_deg)
    loss_deg_mean = loss_deg.mean()
    loss_deg_median = loss_deg.median()

    # Calculate percentage of vectors less than 11.25, 22.5, 30 degrees
    percentage_1 = ((loss_deg < 11.25).sum().float() / total_valid_pixels) * 100
    percentage_2 = ((loss_deg < 22.5).sum().float() / total_valid_pixels) * 100
    percentage_3 = ((loss_deg < 30).sum().float() / total_valid_pixels) * 100

    return loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3


def metric_calculator(input_vec, target_vec, mask=None):
    """Calculate mean, median and angle error between prediction and ground truth
    Args:
        input_vec (tensor): The 1st vectors with whom cosine loss is to be calculated
                            The dimensions of are expected to be (3, height, width).
        target_vec (tensor): The 2nd vectors with whom cosine loss is to be calculated.
                             This should be GROUND TRUTH vector.
                             The dimensions are expected to be (3, height, width).
        mask (tensor): Optional mask of area where loss is to be calculated. All other pixels are ignored.
                       Shape: (height, width), dtype=uint8
    Returns:
        float: The mean error in 2 surface normals in degrees
        float: The median error in 2 surface normals in degrees
        float: The percentage of pixels with error less than 11.25 degrees
        float: The percentage of pixels with error less than 22.5 degrees
        float: The percentage of pixels with error less than 3 degrees
    """
    if len(input_vec.shape) != 3:
        raise ValueError('Shape of tensor must be [C, H, W]. Got shape: {}'.format(input_vec.shape))
    if len(target_vec.shape) != 3:
        raise ValueError('Shape of tensor must be [C, H, W]. Got shape: {}'.format(target_vec.shape))

    INVALID_PIXEL_VALUE = 0  # All 3 channels should have this value
    mask_valid_pixels = ~(torch.all(target_vec == INVALID_PIXEL_VALUE, dim=0))
    if mask is not None:
        mask_valid_pixels = (mask_valid_pixels.float() * mask).byte()
    total_valid_pixels = mask_valid_pixels.sum()
    # TODO: How to deal with a case with zero valid pixels?
    if (total_valid_pixels == 0):
        print('[WARN]: Image found with ZERO valid pixels to calc metrics')
        return torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), mask_valid_pixels

    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    loss_cos = cos(input_vec, target_vec)

    # Taking torch.acos() of 1 or -1 results in NaN. We avoid this by small value epsilon.
    eps = 1e-10
    loss_cos = torch.clamp(loss_cos, (-1.0 + eps), (1.0 - eps))
    loss_rad = torch.acos(loss_cos)
    loss_deg = loss_rad * (180.0 / math.pi)

    # Mask out all invalid pixels and calc mean, median
    temp = loss_deg[0,:,:,:]
    loss_deg = loss_deg[mask_valid_pixels]
    loss_deg_mean = loss_deg.mean()
    loss_deg_median = loss_deg.median()

    # Calculate percentage of vectors less than 11.25, 22.5, 30 degrees
    percentage_1 = ((loss_deg < 11.25).sum().float() / total_valid_pixels) * 100
    percentage_2 = ((loss_deg < 22.5).sum().float() / total_valid_pixels) * 100
    percentage_3 = ((loss_deg < 30).sum().float() / total_valid_pixels) * 100

    return loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3, mask_valid_pixels


# TODO: Fix the loss func to ignore invalid pixels
def loss_fn_radians(input_vec, target_vec, reduction='sum'):
    '''Loss func for estimation of surface normals. Calculated the angle between 2 vectors
    by taking the inverse cos of cosine loss.
    Arguments:
        input_vec {tensor} -- First vector with whole loss is to be calculated.
                              Expected size (batchSize, 3, height, width)
        target_vec {tensor} -- Second vector with whom the loss is to be calculated.
                               Expected size (batchSize, 3, height, width)
    Keyword Arguments:
        reduction {str} -- Can have values 'elementwise_mean' and 'none'.
                           If 'elemtwise_mean' is passed, the mean of all elements is returned
                           if 'none' is passed, a matrix of all cosine losses is returned, same size as input.
                           (default: {'elementwise_mean'})
    Raises:
        Exception -- If any unknown value passed as reduction argument.
    Returns:
        tensor -- Loss from 2 input vectors. Size depends on value of reduction arg.
    '''

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    loss_cos = cos(input_vec, target_vec)
    loss_rad = torch.acos(loss_cos)
    if reduction == 'elementwise_mean':
        loss_rad = torch.mean(loss_rad)
    elif reduction == 'sum':
        loss_rad = torch.sum(loss_rad)
    elif reduction == 'none':
        pass
    else:
        raise Exception(
            'Invalid value for reduction  parameter passed. Please use \'elementwise_mean\' or \'none\''.format())

    return loss_rad


def cross_entropy2d(logit, target, ignore_index=255, weight=None, batch_average=True):
    """
    The loss is
    .. math::
        \sum_{i=1}^{\\infty} x_{i}
        `(minibatch, C, d_1, d_2, ..., d_K)`
    Args:
        logit (Tensor): Output of network
        target (Tensor): Ground Truth
        ignore_index (int, optional): Defaults to 255. The pixels with this labels do not contribute to loss
        weight (List, optional): Defaults to None. Weight assigned to each class
        batch_average (bool, optional): Defaults to True. Whether to consider the loss of each element in the batch.
    Returns:
        Float: The value of loss.
    """

    n, c, h, w = logit.shape
    target = target.squeeze(1)

    if weight is None:
        criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='sum')
    else:
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weight, dtype=torch.float32),
                                        ignore_index=ignore_index,
                                        reduction='sum')

    loss = criterion(logit, target.long())

    if batch_average:
        loss /= n

    return loss