# -*-coding:utf-8-*-

import os

from termcolor import colored

import torch
import torch.nn as nn
from imgaug import augmenters as iaa
from torch.utils.data import DataLoader,SubsetRandomSampler
from torchvision import transforms
from tqdm import tqdm
from modeling import deeplab
import dataloader_real
import loss_functions
import numpy as np
import API.utils
###################### DataLoader #############################

#-- 1、 config parameters


imgHeight = 512
imgWidth = 512
batch_size = 8
num_workers = 0
validation_split = 0.1
shuffle_dataset = True
pin_memory = False
prefetch_factor = 2

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
######## train dataset concat ########
dataset_tiny_white_cup_11_28 = dataloader_real.RealSurfaceNormalsDataset(input_I_sum_dir='/media/zjw/data/smq/samples/End2End/Tiny-White-Cup-11-28/I-sum',
                                                                         dolp_dir= '/media/zjw/data/smq/samples/End2End/Tiny-White-Cup-11-28/params/DoLP',
                                                                         aolp_dir= '/media/zjw/data/smq/samples/End2End/Tiny-White-Cup-11-28/params/AoLP',
                                                                         mask_dir = '/media/zjw/data/smq/samples/End2End/Tiny-White-Cup-11-28/masks',
                                                                         label_dir= '/media/zjw/data/smq/samples/End2End/Tiny-White-Cup-11-28/normals-png',transform=augs_train)
dataset_tiny_white_cup_11_30 = dataloader_real.RealSurfaceNormalsDataset(input_I_sum_dir='/media/zjw/data/smq/samples/End2End/Tiny-White-Cup-11-30/I-sum',
                                                                         dolp_dir = '/media/zjw/data/smq/samples/End2End/Tiny-White-Cup-11-30/params/DoLP',
                                                                         aolp_dir = '/media/zjw/data/smq/samples/End2End/Tiny-White-Cup-11-30/params/AoLP',
                                                                         mask_dir = '/media/zjw/data/smq/samples/End2End/Tiny-White-Cup-11-30/masks',
                                                                         label_dir= '/media/zjw/data/smq/samples/End2End/Tiny-White-Cup-11-30/normals-png',transform=augs_train)
dataset_plastic_cup_11_28 = dataloader_real.RealSurfaceNormalsDataset(input_I_sum_dir='/media/zjw/data/smq/samples/End2End/Plastic-Cup-11-28/I-sum',
                                                                      dolp_dir = '/media/zjw/data/smq/samples/End2End/Plastic-Cup-11-28/params/DoLP',
                                                                      aolp_dir= '/media/zjw/data/smq/samples/End2End/Plastic-Cup-11-28/params/AoLP',
                                                                      mask_dir='/media/zjw/data/smq/samples/End2End/Plastic-Cup-11-28/masks',
                                                                      label_dir= '/media/zjw/data/smq/samples/End2End/Plastic-Cup-11-28/normals-png',transform=augs_train)
dataset_plastic_cup_11_30 = dataloader_real.RealSurfaceNormalsDataset(input_I_sum_dir='/media/zjw/data/smq/samples/End2End/Plastic-Cup-11-30/I-sum',
                                                                      dolp_dir = '/media/zjw/data/smq/samples/End2End/Plastic-Cup-11-30/params/DoLP',
                                                                      aolp_dir = '/media/zjw/data/smq/samples/End2End/Plastic-Cup-11-30/params/AoLP',
                                                                         mask_dir='/media/zjw/data/smq/samples/End2End/Plastic-Cup-11-30/masks',
                                                                         label_dir= '/media/zjw/data/smq/samples/End2End/Plastic-Cup-11-30/normals-png',transform=augs_train)
dataset_middle_round_cup_11_28 = dataloader_real.RealSurfaceNormalsDataset(input_I_sum_dir='/media/zjw/data/smq/samples/End2End/Middle-Round-Cup-11-28/I-sum',
                                                                           dolp_dir = '/media/zjw/data/smq/samples/End2End/Middle-Round-Cup-11-28/params/DoLP',
                                                                           aolp_dir = '/media/zjw/data/smq/samples/End2End/Middle-Round-Cup-11-28/params/AoLP',
                                                                         mask_dir= '/media/zjw/data/smq/samples/End2End/Middle-Round-Cup-11-28/masks',
                                                                         label_dir= '/media/zjw/data/smq/samples/End2End/Middle-Round-Cup-11-28/normals-png',transform=augs_train)
dataset_middle_round_cup_11_30 = dataloader_real.RealSurfaceNormalsDataset(input_I_sum_dir='/media/zjw/data/smq/samples/End2End/Middle-Round-Cup-11-30/I-sum',
                                                                           dolp_dir = '/media/zjw/data/smq/samples/End2End/Middle-Round-Cup-11-30/params/DoLP',
                                                                           aolp_dir = '/media/zjw/data/smq/samples/End2End/Middle-Round-Cup-11-30/params/AoLP',
                                                                         mask_dir = '/media/zjw/data/smq/samples/End2End/Middle-Round-Cup-11-30/masks',
                                                                         label_dir= '/media/zjw/data/smq/samples/End2End/Middle-Round-Cup-11-30/normals-png',transform=augs_train)
dataset_little_square_cup_11_28 = dataloader_real.RealSurfaceNormalsDataset(input_I_sum_dir='/media/zjw/data/smq/samples/End2End/Little-Square-Cup-11-28/I-sum',
                                                                            dolp_dir = '/media/zjw/data/smq/samples/End2End/Little-Square-Cup-11-28/params/DoLP',
                                                                            aolp_dir = '/media/zjw/data/smq/samples/End2End/Little-Square-Cup-11-28/params/AoLP',
                                                                         mask_dir = '/media/zjw/data/smq/samples/End2End/Little-Square-Cup-11-28/masks',
                                                                         label_dir= '/media/zjw/data/smq/samples/End2End/Little-Square-Cup-11-28/normals-png',transform=augs_train)
dataset_little_square_cup_11_30 = dataloader_real.RealSurfaceNormalsDataset(input_I_sum_dir='/media/zjw/data/smq/samples/End2End/Little-Square-Cup-11-30/I-sum',
                                                                            dolp_dir = '/media/zjw/data/smq/samples/End2End/Little-Square-Cup-11-30/params/DoLP',
                                                                            aolp_dir = '/media/zjw/data/smq/samples/End2End/Little-Square-Cup-11-30/params/AoLP',
                                                                         mask_dir= '/media/zjw/data/smq/samples/End2End/Little-Square-Cup-11-30/masks',
                                                                         label_dir= '/media/zjw/data/smq/samples/End2End/Little-Square-Cup-11-30/normals-png',transform=augs_train)
db_list = [dataset_tiny_white_cup_11_28,dataset_tiny_white_cup_11_30,dataset_plastic_cup_11_28,dataset_plastic_cup_11_30,dataset_middle_round_cup_11_28,dataset_middle_round_cup_11_30,dataset_little_square_cup_11_28,dataset_little_square_cup_11_30]
dataset = torch.utils.data.ConcatDataset(db_list)


#-- 2、create dataloader
# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
random_seed = 42
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
trainLoader = DataLoader(dataset,
                         num_workers=num_workers,
                         batch_size=batch_size,
                         drop_last=True,
                         pin_memory=pin_memory,sampler=train_sampler)
testLoader = DataLoader(dataset,
                         batch_size=1,
                         num_workers=num_workers,
                         drop_last=True,
                         pin_memory=pin_memory,sampler=valid_sampler)


print("trainLoader size:",trainLoader.__len__()*trainLoader.batch_size)
print("testLoader size:",testLoader.__len__()*testLoader.batch_size)

###################### ModelBuilder #############################

#-- 1、 config parameters
# backbone_model = 'resnet50'
backbone_model = 'mobilenet'
sync_bn = False  # this is for Multi-GPU synchronize batch normalization
numClasses = 3

#-- 2、create model
model = deeplab.DeepLab(num_classes=numClasses,
                        backbone=backbone_model,
                        sync_bn=sync_bn,
                        freeze_bn=False)

#-- 3、Enable GPU for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
model = model.to(device)


###################### Setup Optimizer #############################


#-- 1、 config parameters
learningRate = 1e-6
weightDecay = 5e-4
momentum = 0.9
# lrSchedulerStep
lrScheduler = 'StepLR'
step_size = 7
gamma = 0.1
# lrSchedulerPlateau:
factor: 0.8
patience: 25
verbose: True
# loss func


#-- 2、create optimizer
optimizer = torch.optim.SGD(model.parameters(),
                            lr=float(learningRate),
                            momentum=float(momentum),
                            weight_decay=float(weightDecay))

#-- 3、create learningRate schduler
if not lrScheduler:
    pass
elif lrScheduler == 'StepLR':
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=step_size,
                                                   gamma=float(gamma))
elif lrScheduler == 'ReduceLROnPlateau':
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                              factor=float(factor),
                                                              patience=patience,
                                                              verbose=verbose)
elif lrScheduler == 'lr_poly':
    pass
else:
    raise ValueError(
        "Invalid Scheduler from config file: '{}'. Valid values are ['', 'StepLR', 'ReduceLROnPlateau']".format(
            lrScheduler))

#-- 4、select koss func
criterion = loss_functions.loss_fn_cosine


###################### Train Model #############################
#-- 1、config parameters

#-- 2、load check point
CHECKPOINT_DIR = '/home/zjw/smq/SurfaceNormals/CheckPoints'
checkpoint_path = os.path.join(CHECKPOINT_DIR,'check-point-epoch-0099.pth')

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

model.eval() # eval mode, freeze params
running_loss = 0.0
running_mean = 0
running_median = 0
model.eval()
for iter_num, sample_batched in enumerate(tqdm(testLoader)):
    # print('')
    inputs_t, label_t,mask_t = sample_batched
    inputs_t = inputs_t.to(device)
    label_t = label_t.to(device)

    with torch.no_grad():
        normal_vectors = model(inputs_t)

    normal_vectors_norm = nn.functional.normalize(normal_vectors.double(), p=2, dim=1)
    loss = criterion(normal_vectors_norm, label_t.double(),reduction='sum')
    loss /= batch_size
    # calcute metrics
    inputs_t = inputs_t.detach().cpu()
    label_t = label_t.detach().cpu()
    normal_vectors_norm = normal_vectors_norm.detach().cpu()

    loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3 = loss_functions.metric_calculator_batch(
        normal_vectors_norm, label_t.double())
    running_mean += loss_deg_mean.item()
    running_median += loss_deg_median.item()
    running_loss += loss.item()

    # save validation pictures
    label_t_rgb = label_t.numpy().squeeze(0).transpose(1, 2, 0)
    label_t_rgb = API.utils.normal_to_rgb(label_t_rgb)
    predict_norm = normal_vectors_norm.numpy().squeeze(0).transpose(1, 2, 0)
    mask_t = mask_t.squeeze(1)
    predict_norm[mask_t.squeeze(0) == 0, :] = -1
    predict_norm_rgb = API.utils.normal_to_rgb(predict_norm)
    API.utils.png_saver(os.path.join('/home/zjw/smq/SurfaceNormals/results', str(iter_num).zfill(3) + '-label.png'),
                        label_t_rgb)
    API.utils.png_saver(os.path.join('/home/zjw/smq/SurfaceNormals/results', str(iter_num).zfill(3) + '-predict.png'),
                        predict_norm_rgb)
    print(iter_num,"test running loss:", loss.item())
    print(iter_num,"test running mean:", loss_deg_mean.item())
    print(iter_num,"test running median:", loss_deg_median.item())

num_samples = len(testLoader)
print("test running loss:",running_loss/num_samples)
print("test running mean:",running_mean/num_samples)
print("test running median:",running_median/num_samples)