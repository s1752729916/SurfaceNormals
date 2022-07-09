# -*-coding:utf-8-*-

import os

from termcolor import colored
import torch
import torch.nn as nn
from imgaug import augmenters as iaa
from torch.utils.data import DataLoader,SubsetRandomSampler
from tqdm import tqdm
from modeling import deeplab
from dataloader import dataloaderIDA,dataloaderI,dataloaderIDAN,dataloaderDA,dataloladerN,dataloaderDAN,datasetSfP
from modeling.PS_FCN.PS_FCN import PS_FCN
from modeling.smqFusion.smqFusion import smqFusion
import loss_functions
import API.utils
import numpy as np
import random
from evaluation import evaluation
from tensorboardX import SummaryWriter

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(20)

###################### DataLoader #############################

#-- 1、 config parameters


imgHeight = 512
imgWidth = 512
batch_size = 4
num_workers = 2
validation_split = 0.1
shuffle_dataset = True
pin_memory = False
prefetch_factor = 8

    #-- 2、create dataset
augs_train = iaa.Sequential([
    # Geometric Augs
    iaa.Resize({"height": imgHeight, "width": imgWidth }, interpolation='nearest'),  # Resize image
    # iaa.Fliplr(0.5),
    # iaa.Flipud(0.5),
    # iaa.Rot90((0, 4)),
    # Blur and Noise
    # iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 1.5), name="gaus-blur")),x`x
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
dataset_train = datasetSfP.SfPSurfaceDataset(root_dir = '/media/zjw/smq/data/DeepSfP/pngs',list_file='/media/zjw/smq/data/DeepSfP/pngs/train_list.csv',transform=augs_train)
dataset_test = datasetSfP.SfPSurfaceDataset(root_dir = '/media/zjw/smq/data/DeepSfP/pngs',list_file='/media/zjw/smq/data/DeepSfP/pngs/test_list.csv',transform=augs_train)




#-- 2、create dataloader
# Creating data indices for training and validation splits:
dataset_size = len(dataset_train)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
random_seed = 42
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
trainLoader = DataLoader(dataset_train,
                         num_workers=num_workers,
                         batch_size=batch_size,
                         drop_last=True,
                         pin_memory=pin_memory)
testLoader = DataLoader(dataset_test,
                         num_workers=num_workers,
                         batch_size=1,
                         drop_last=True,
                         pin_memory=pin_memory)




print("trainLoader size:",trainLoader.__len__()*trainLoader.batch_size)

###################### ModelBuilder #############################

#-- 1、 config parameters
# backbone_model = 'resnet50'
backbone_model = 'resnet50'
sync_bn = False  # this is for Multi-GPU synchronize batch normalization
numClasses = 3
use_atten = False

#-- 2、create model

# model = deeplab.DeepLab(num_classes=numClasses,
#                         backbone=backbone_model,
#                         sync_bn=sync_bn,
#                         freeze_bn=False)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = smqFusion(backbone = backbone_model,num_classes=3,device=device,sync_bn=False)
#-- 3、Enable GPU for training

import os
# device = torch.device("cpu")
# model = model.to(device)
model = model.to(device)


###################### Setup Optimizer #############################


#-- 1、 config parameters
learningRate = 0.01
weightDecay = 5e-4
momentum = 0.9
# lrSchedulerStep
lrScheduler = 'StepLR'
step_size = 10
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

#-- 4、select koss fu
criterion = loss_functions.my_loss_cosine
writer = SummaryWriter()


###################### Train Model #############################
#-- 1、config parameters
MAX_EPOCH = 22
saveModelInterval = 10
CHECKPOINT_DIR = '/home/zjw/smq/project/SurfaceNormals/SfPDataset/checkpoints'
total_iter_num = 0
START_EPOCH = 0
continue_train = False
preCheckPoint = os.path.join(CHECKPOINT_DIR,'check-point-epoch-0000.pth')

#-- 2、load check point
if(continue_train):
    print(colored('Continuing training from checkpoint...Loaded data from checkpoint:', 'green'))
    if not os.path.isfile(preCheckPoint):
        raise ValueError('Invalid path to the given weights file for transfer learning.\
                The file {} does not exist'.format(preCheckPoint))
    CHECKPOINT = torch.load(preCheckPoint, map_location='cpu')
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

    if continue_train and preCheckPoint:
        if 'optimizer_state_dict' in CHECKPOINT:
            optimizer.load_state_dict(CHECKPOINT['optimizer_state_dict'])
        else:
            print(
                colored(
                    'WARNING: Could not load optimizer state from checkpoint as checkpoint does not contain ' +
                    '"optimizer_state_dict". Continuing without loading optimizer state. ', 'red'))
    if continue_train and preCheckPoint:
        if 'model_state_dict' in CHECKPOINT:
            # TODO: remove this second check for 'model_state_dict' soon. Kept for ensuring backcompatibility
            total_iter_num = CHECKPOINT['total_iter_num'] + 1
            START_EPOCH = CHECKPOINT['epoch'] + 1
            END_EPOCH = CHECKPOINT['epoch'] + MAX_EPOCH
        else:
            print(
                colored(
                    'Could not load epoch and total iter nums from checkpoint, they do not exist in checkpoint.\
                           Starting from epoch num 0', 'red'))
#-- 3、epoch cycle
import time
mean_list = []
median_list = []
for epoch in range(START_EPOCH,MAX_EPOCH):
    print('\n\nEpoch {}/{}'.format(epoch, MAX_EPOCH - 1))
    print('-' * 30)

    ###################### Training Cycle #############################
    print('Train:')
    print('=' * 10)
    model.train()  # set model mode to train mode

    running_loss = 0.0
    running_mean = 0
    running_median = 0
    for iter_num,batch  in enumerate(tqdm(trainLoader)):
        total_iter_num+=1
        params_t,normals_t, label_t,mask_t = batch
        params_t = params_t.to(device)
        normals_t = normals_t.to(device)
        label_t = label_t.to(device)
        # Forward + Backward Prop
        start = time.time()
        optimizer.zero_grad()
        torch.set_grad_enabled(True)
        normal_vectors,atten_map = model(params_t,normals_t)
        normal_vectors_norm = nn.functional.normalize(normal_vectors.double(), p=2, dim=1)
        normal_vectors_norm = normal_vectors_norm
        loss = criterion(normal_vectors_norm, label_t.double(),mask_tensor=mask_t,atten_map = atten_map,reduction='elementwise_mean',device=device,use_atten = use_atten)
        loss /= batch_size
        loss.backward()
        optimizer.step()
        # print('time consume:',time.time()-start)

        # calcute metrics
        label_t = label_t.detach().cpu()
        normal_vectors_norm = normal_vectors_norm.detach().cpu()

        loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3 = loss_functions.metric_calculator_batch(
            normal_vectors_norm.detach().cpu(), label_t.double())
        running_mean += loss_deg_mean.item()
        running_median += loss_deg_median.item()
        running_loss += loss.item()

        #  output train set
        if(epoch % 10==0):
            label_t_rgb = label_t.numpy()[0,:,:,:].transpose(1, 2, 0)
            label_t_rgb = API.utils.normal_to_rgb(label_t_rgb)
            predict_norm = normal_vectors_norm.numpy()[0,:,:,:].transpose(1, 2, 0)
            mask_t = mask_t.squeeze(1)
            predict_norm[mask_t[0,:,:] == 0, :] = -1
            predict_norm_rgb = API.utils.normal_to_rgb(predict_norm)
            atten_map = atten_map[0,:,:,:]
            atten_map_rgb = atten_map.detach().cpu().numpy().transpose(1, 2, 0)

            atten_map_rgb = atten_map_rgb * 255
            atten_map_rgb = atten_map_rgb.astype(np.uint8)
            API.utils.png_saver(
                os.path.join('/home/zjw/smq/project/SurfaceNormals/SfPDataset/results/train', str(iter_num).zfill(3) + '-label.png'),
                label_t_rgb)
            API.utils.png_saver(
                os.path.join('/home/zjw/smq/project/SurfaceNormals/SfPDataset/results/train', str(iter_num).zfill(3) + '-predict.png'),
                predict_norm_rgb)
            API.utils.png_saver(
                os.path.join('/home/zjw/smq/project/SurfaceNormals/SfPDataset/results/train', str(iter_num).zfill(3) + '-atten.png'),
                atten_map_rgb)


        # print('loss_deg_mean:',loss_deg_mean)
        # print('loss_deg_median:',loss_deg_median)
    num_samples = (len(trainLoader))
    epoch_loss = running_loss/num_samples
    print("train running loss:",epoch_loss)
    print("train running mean:",running_mean/num_samples)
    print("train running median:",running_median/num_samples)


    # save the model check point every N epoch
    if epoch % saveModelInterval==0:
        filename = os.path.join(CHECKPOINT_DIR,'check-point-epoch-{:04d}.pth'.format(epoch))
        model_params = model.state_dict()
        torch.save(
            {
                'model_state_dict': model_params,
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'total_iter_num': total_iter_num,
                'epoch_loss': epoch_loss,
            }, filename)



    ###################### Validation Cycle #############################
    mean_all = 0
    median_all = 0
    acc_all_1 = 0
    acc_all_2 = 0
    acc_all_3 = 0
    count = 0
    print('\nValidation:')
    print('=' * 10)
    running_loss,running_mean,running_median,running_percentage_1,running_percentage_2,running_percentage_3 = evaluation(model = model,
            testLoader= testLoader,device=device,criterion=criterion,use_atten=use_atten,epoch = epoch,name = 'deepSfP_test',writer=writer,resultPath='/home/zjw/smq/project/SurfaceNormals/SfPDataset/results/test')
    print('loss: ',running_loss)
    print('mean: ',running_mean)
    print('median: ',running_median)
    print('percentage_1: ',running_percentage_1)
    print('percentage_2: ',running_percentage_2)
    print('percentage_3: ',running_percentage_3)
    print('=' * 10)
    print('\n')





