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
import dataloader
import loss_functions


###################### DataLoader #############################

#-- 1、 config parameters


imgHeight = 512
imgWidth = 512
batch_size = 8
num_workers = 4
validation_split = 0.1
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
######## train dataset concat ########
dataset_train_cup_with_waves = dataloader.SurfaceNormalsDataset(
    input_rgb_dir='/media/smq/ClearGrasp/cleargrasp-dataset-train/cup-with-waves-train/rgb-imgs',
    input_normal_dir='/media/smq/ClearGrasp/cleargrasp-dataset-train/cup-with-waves-train/synthesis-normals',
    label_dir='/media/smq/ClearGrasp/cleargrasp-dataset-train/cup-with-waves-train/camera-normals',
    mask_dir='/media/smq/ClearGrasp/cleargrasp-dataset-train/cup-with-waves-train/segmentation-masks',
    transform=augs_train, input_only=input_only)
dataset_train_flower_bath_bomb = dataloader.SurfaceNormalsDataset(
    input_rgb_dir='/media/smq/ClearGrasp/cleargrasp-dataset-train/flower-bath-bomb-train/rgb-imgs',
    input_normal_dir='/media/smq/ClearGrasp/cleargrasp-dataset-train/flower-bath-bomb-train/synthesis-normals',
    label_dir='/media/smq/ClearGrasp/cleargrasp-dataset-train/flower-bath-bomb-train/camera-normals',
    mask_dir='/media/smq/ClearGrasp/cleargrasp-dataset-train/flower-bath-bomb-train/segmentation-masks',
    transform=augs_train, input_only=input_only)
dataset_train_heart_bath_bomb = dataloader.SurfaceNormalsDataset(
    input_rgb_dir='/media/smq/ClearGrasp/cleargrasp-dataset-train/heart-bath-bomb-train/rgb-imgs/',
    input_normal_dir='/media/smq/ClearGrasp/cleargrasp-dataset-train/heart-bath-bomb-train/synthesis-normals',
    label_dir='/media/smq/ClearGrasp/cleargrasp-dataset-train/heart-bath-bomb-train/camera-normals',
    mask_dir='/media/smq/ClearGrasp/cleargrasp-dataset-train/heart-bath-bomb-train/segmentation-masks',
    transform=augs_train, input_only=input_only)
dataset_train_square_plastic_bottle = dataloader.SurfaceNormalsDataset(
    input_rgb_dir='/media/smq/ClearGrasp/cleargrasp-dataset-train/square-plastic-bottle-train/rgb-imgs',
    input_normal_dir='/media/smq/ClearGrasp/cleargrasp-dataset-train/square-plastic-bottle-train/synthesis-normals',
    label_dir='/media/smq/ClearGrasp/cleargrasp-dataset-train/square-plastic-bottle-train/camera-normals',
    mask_dir='/media/smq/ClearGrasp/cleargrasp-dataset-train/square-plastic-bottle-train/segmentation-masks',
    transform=augs_train, input_only=input_only)
dataset_train_stemless_plastic_champagne_glass = dataloader.SurfaceNormalsDataset(
    input_rgb_dir='/media/smq/ClearGrasp/cleargrasp-dataset-train/stemless-plastic-champagne-glass-train/rgb-imgs',
    input_normal_dir='/media/smq/ClearGrasp/cleargrasp-dataset-train/stemless-plastic-champagne-glass-train/synthesis-normals',
    label_dir='/media/smq/ClearGrasp/cleargrasp-dataset-train/stemless-plastic-champagne-glass-train/camera-normals',
    mask_dir='/media/smq/ClearGrasp/cleargrasp-dataset-train/stemless-plastic-champagne-glass-train/segmentation-masks',
    transform=augs_train, input_only=input_only)

######## test dataset concat #########
dataset_test_glass_round_potion = dataloader.SurfaceNormalsDataset(
    input_rgb_dir='/media/smq/ClearGrasp/cleargrasp-dataset-test-val/synthetic-test/glass-round-potion-test/rgb-imgs',
    input_normal_dir='/media/smq/ClearGrasp/cleargrasp-dataset-test-val/synthetic-test/glass-round-potion-test/synthesis-normals',
    label_dir='/media/smq/ClearGrasp/cleargrasp-dataset-test-val/synthetic-test/glass-round-potion-test/camera-normals',
    mask_dir='/media/smq/ClearGrasp/cleargrasp-dataset-test-val/synthetic-test/glass-round-potion-test/segmentation-masks',
    transform=augs_train, input_only=input_only
)
dataset_test_glass_square_potion = dataloader.SurfaceNormalsDataset(
    input_rgb_dir='/media/smq/ClearGrasp/cleargrasp-dataset-test-val/synthetic-test/glass-square-potion-test/rgb-imgs',
    input_normal_dir='/media/smq/ClearGrasp/cleargrasp-dataset-test-val/synthetic-test/glass-square-potion-test/synthesis-normals',
    label_dir='/media/smq/ClearGrasp/cleargrasp-dataset-test-val/synthetic-test/glass-square-potion-test/camera-normals',
    mask_dir='/media/smq/ClearGrasp/cleargrasp-dataset-test-val/synthetic-test/glass-square-potion-test/segmentation-masks',
    transform=augs_train, input_only=input_only
)
dataset_test_star_bath_bomb = dataloader.SurfaceNormalsDataset(
    input_rgb_dir='/media/smq/ClearGrasp/cleargrasp-dataset-test-val/synthetic-test/star-bath-bomb-test/rgb-imgs',
    input_normal_dir='/media/smq/ClearGrasp/cleargrasp-dataset-test-val/synthetic-test/star-bath-bomb-test/synthesis-normals',
    label_dir='/media/smq/ClearGrasp/cleargrasp-dataset-test-val/synthetic-test/star-bath-bomb-test/camera-normals',
    mask_dir='/media/smq/ClearGrasp/cleargrasp-dataset-test-val/synthetic-test/star-bath-bomb-test/segmentation-masks',
    transform=augs_train, input_only=input_only
)
dataset_test_tree_bath_bomb = dataloader.SurfaceNormalsDataset(
    input_rgb_dir='/media/smq/ClearGrasp/cleargrasp-dataset-test-val/synthetic-test/tree-bath-bomb-test/rgb-imgs',
    input_normal_dir='/media/smq/ClearGrasp/cleargrasp-dataset-test-val/synthetic-test/tree-bath-bomb-test/synthesis-normals',
    label_dir='/media/smq/ClearGrasp/cleargrasp-dataset-test-val/synthetic-test/tree-bath-bomb-test/camera-normals',
    mask_dir='/media/smq/ClearGrasp/cleargrasp-dataset-test-val/synthetic-test/tree-bath-bomb-test/segmentation-masks',
    transform=augs_train, input_only=input_only
)

db_train_list = [dataset_train_cup_with_waves,dataset_train_flower_bath_bomb,dataset_train_heart_bath_bomb,dataset_train_heart_bath_bomb,dataset_train_square_plastic_bottle,dataset_train_stemless_plastic_champagne_glass]
db_test_list = [dataset_test_glass_round_potion,dataset_test_glass_square_potion,dataset_test_star_bath_bomb,dataset_test_tree_bath_bomb]

dataset_train = torch.utils.data.ConcatDataset(db_train_list)
dataset_test = torch.utils.data.ConcatDataset(db_test_list)
print("number of train dataset: ",dataset_train.__len__())
print("number of test dataset:",dataset_test.__len__())
#-- 2、create dataloader
# Creating data indices for training and validation splits:
# dataset_size = len(dataset)
# indices = list(range(dataset_size))
# split = int(np.floor(validation_split * dataset_size))
# random_seed = 42
# if shuffle_dataset :
#     np.random.seed(random_seed)
#     np.random.shuffle(indices)
# train_indices, val_indices = indices[split:], indices[:split]
#
# train_sampler = SubsetRandomSampler(train_indices)
# valid_sampler = SubsetRandomSampler(val_indices)
# trainLoader = DataLoader(dataset,
#                          num_workers=num_workers,
#                          batch_size=batch_size,
#                          drop_last=True,
#                          pin_memory=pin_memory,sampler=train_sampler,prefetch_factor  = prefetch_factor)
# testLoader = DataLoader(dataset,
#                          batch_size=batch_size,
#                          num_workers=num_workers,
#                          drop_last=True,
#                          pin_memory=pin_memory,sampler=valid_sampler,prefetch_factor = prefetch_factor)

trainLoader = DataLoader(dataset_train,
                         num_workers=num_workers,
                         batch_size=batch_size,
                         drop_last=True,
                         pin_memory=pin_memory,
                         prefetch_factor=prefetch_factor
                         )
testLoader = DataLoader(dataset_test,
                         num_workers=num_workers,
                         batch_size=batch_size,
                         drop_last=True,
                         pin_memory=pin_memory,
                         prefetch_factor=prefetch_factor
                         )
print("trainLoader size:",trainLoader.__len__()*trainLoader.batch_size)
print("testLoader size:",testLoader.__len__()*testLoader.batch_size)

###################### ModelBuilder #############################

#-- 1、 config parameters
backbone_model = 'resnet50'

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
MAX_EPOCH = 50
saveModelInterval = 1
CHECKPOINT_DIR = '/home/zjw/smq/SurfaceNormals/CheckPoints'
total_iter_num = 0
START_EPOCH = 0
continue_train = False
preCheckPoint = os.path.join(CHECKPOINT_DIR,'check-point-epoch-0002.pth')

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
        print('')
        total_iter_num+=1
        inputs_t, label_t,mask_t = batch
        inputs_t = inputs_t.to(device)
        label_t = label_t.to(device)
        # Forward + Backward Prop
        start = time.time()
        optimizer.zero_grad()
        torch.set_grad_enabled(True)
        normal_vectors = model(inputs_t)
        normal_vectors_norm = nn.functional.normalize(normal_vectors.double(), p=2, dim=1)
        loss = criterion(normal_vectors_norm, label_t.double(),reduction='sum')
        loss /= batch_size
        loss.backward()
        optimizer.step()
        # print('time consume:',time.time()-start)

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
        print('loss_deg_mean:',loss_deg_mean)
        print('loss_deg_median:',loss_deg_median)
    num_samples = (len(trainLoader))
    epoch_loss = running_loss/num_samples
    print("train running loss:",epoch_loss)
    print("train running mean:",running_mean/num_samples)
    print("train running median:",running_median/num_samples)


    # save the model check point every N epoch
    if epoch % saveModelInterval==0:
        filename = os.path.join(CHECKPOINT_DIR,'check-point-epoch-{:04d}.pth'.format(epoch))
        if torch.cuda.device_count() > 1:
            model_params = model.module.state_dict()  # Saving nn.DataParallel model
        else:
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
    print('\nValidation:')
    print('=' * 10)

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


    num_samples = len(testLoader)
    print("test running loss:",running_loss/num_samples)
    print("test running mean:",running_mean/num_samples)
    print("test running median:",running_median/num_samples)