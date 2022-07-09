# -*-coding:utf-8-*-
import os
import torch
import torch.nn as nn
from imgaug import augmenters as iaa
from torch.utils.data import DataLoader,SubsetRandomSampler
from tqdm import tqdm
from modeling.TransSfP.dataloader import TransSfPDataset
from modeling.TransSfP.model.TransSfP import TranSfPNet
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
batch_size =3
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
root_dir = '/media/disk2/smq_data/samples/TransSfP'
code_root_dir = '/home/robotlab/smq/SurfaceNormals'
dataset_middle_round_cup_black_background_12_28 = TransSfPDataset(dolp_dir = root_dir + '/real-world/middle-round-cup/params/DoLP',
                                                                                  aolp_dir = root_dir + '/real-world/middle-round-cup/params/AoLP',
                                                                                  synthesis_normals_dir= root_dir + '/real-world/middle-round-cup/synthesis-normals',
                                                                                  mask_dir= root_dir + '/real-world/middle-round-cup/masks',
                                                                                  label_dir= root_dir + '/real-world/middle-round-cup/normals-png', transform=augs_train)

dataset_middle_square_cup_black_background_12_28 = TransSfPDataset(
    dolp_dir= root_dir + '/real-world/middle-square-cup/params/DoLP',
    aolp_dir= root_dir + '/real-world/middle-square-cup/params/AoLP',
    synthesis_normals_dir= root_dir + '/real-world/middle-square-cup/synthesis-normals',
                                                                                   mask_dir= root_dir + '/real-world/middle-square-cup/masks',
                                                                                   label_dir= root_dir + '/real-world/middle-square-cup/normals-png', transform=augs_train)

dataset_middle_white_cup_black_background_12_28 = TransSfPDataset(
    dolp_dir= root_dir + '/real-world/middle-white-cup/params/DoLP',
    aolp_dir= root_dir + '/real-world/middle-white-cup/params/AoLP',
    synthesis_normals_dir= root_dir + '/real-world/middle-white-cup/synthesis-normals',
                                                                                  mask_dir= root_dir + '/real-world/middle-white-cup/masks',
                                                                                  label_dir= root_dir + '/real-world/middle-white-cup/normals-png', transform=augs_train)



dataset_tiny_white_cup_black_background_12_28 = TransSfPDataset(
    dolp_dir= root_dir + '/real-world/tiny-cup/params/DoLP',
    aolp_dir= root_dir + '/real-world/tiny-cup/params/AoLP',
    synthesis_normals_dir= root_dir + '/real-world/tiny-cup/synthesis-normals',
                                                                                mask_dir= root_dir + '/real-world/tiny-cup/masks',
                                                                                label_dir= root_dir + '/real-world/tiny-cup/normals-png', transform=augs_train)

dataset_tiny_white_cup_edges_black_background_12_28 = TransSfPDataset(
    dolp_dir= root_dir + '/real-world/tiny-cup-edges/params/DoLP',
    aolp_dir= root_dir + '/real-world/tiny-cup-edges/params/AoLP',
    synthesis_normals_dir=  root_dir + '/real-world/tiny-cup-edges/synthesis-normals',
                                                                                      mask_dir = root_dir + '/real-world/tiny-cup-edges/masks',
                                                                                      label_dir= root_dir + '/real-world/tiny-cup-edges/normals-png', transform=augs_train)
dataset_bird_back_1_20 = TransSfPDataset(
    dolp_dir= root_dir + '/real-world/bird-back/params/DoLP',
    aolp_dir= root_dir + '/real-world/bird-back/params/AoLP',
    synthesis_normals_dir= root_dir + '/real-world/bird-back/synthesis-normals',
                                                                                      mask_dir = root_dir + '/real-world/bird-back/masks',
                                                                                      label_dir= root_dir + '/real-world/bird-back/normals-png', transform=augs_train)
dataset_bird_front_1_20 = TransSfPDataset(
    dolp_dir= root_dir + '/real-world/bird-front/params/DoLP',
    aolp_dir= root_dir + '/real-world/bird-front/params/AoLP',
    synthesis_normals_dir= root_dir + '/real-world/bird-front/synthesis-normals',
                                                                                      mask_dir = root_dir + '/real-world/bird-front/masks',
                                                                                      label_dir= root_dir + '/real-world/bird-front/normals-png', transform=augs_train)
dataset_cat_front_1_20 = TransSfPDataset(
    dolp_dir= root_dir + '/real-world/cat-front/params/DoLP',
    aolp_dir= root_dir + '/real-world/cat-front/params/AoLP',
    synthesis_normals_dir= root_dir + '/real-world/cat-front/synthesis-normals',
                                                                                      mask_dir = root_dir + '/real-world/cat-front/masks',
                                                                                      label_dir= root_dir + '/real-world/cat-front/normals-png', transform=augs_train)
dataset_cat_back_1_20 = TransSfPDataset(
    dolp_dir= root_dir + '/real-world/cat-back/params/DoLP',
    aolp_dir= root_dir + '/real-world/cat-back/params/AoLP',
    synthesis_normals_dir= root_dir + '/real-world/cat-back/synthesis-normals',
                                                                                      mask_dir = root_dir + '/real-world/cat-back/masks',
                                                                                      label_dir= root_dir + '/real-world/cat-back/normals-png', transform=augs_train)
dataset_hemi_sphere_big_1_20 = TransSfPDataset(
    dolp_dir= root_dir + '/real-world/hemi-sphere-big/params/DoLP',
    aolp_dir= root_dir + '/real-world/hemi-sphere-big/params/AoLP',
    synthesis_normals_dir= root_dir + '/real-world/hemi-sphere-big/synthesis-normals',
                                                                                      mask_dir = root_dir + '/real-world/hemi-sphere-big/masks',
                                                                                      label_dir= root_dir + '/real-world/hemi-sphere-big/normals-png', transform=augs_train)
dataset_hemi_sphere_small_1_20 = TransSfPDataset(
    dolp_dir= root_dir + '/real-world/hemi-sphere-small/params/DoLP',
    aolp_dir= root_dir + '/real-world/hemi-sphere-small/params/AoLP',
    synthesis_normals_dir= root_dir + '/real-world/hemi-sphere-small/synthesis-normals',
                                                                                      mask_dir = root_dir + '/real-world/hemi-sphere-small/masks',
                                                                                      label_dir= root_dir + '/real-world/hemi-sphere-small/normals-png', transform=augs_train)

# synthetic datasets

dataset_synthetic_polar_bun_zipper_back = TransSfPDataset(
    dolp_dir= root_dir + '/synthetic/bun-zipper-back/params/DoLP',
    aolp_dir= root_dir + '/synthetic/bun-zipper-back/params/AoLP',
    synthesis_normals_dir= root_dir + '/synthetic/bun-zipper-back/synthesis-normals',
                                                                          mask_dir= root_dir + '/synthetic/bun-zipper-back/masks',
                                                                          label_dir= root_dir + '/synthetic/bun-zipper-back/normals-png',
                                                                          transform=augs_train)
dataset_synthetic_polar_bun_zipper_front = TransSfPDataset(
    dolp_dir= root_dir + '/synthetic/bun-zipper-front/params/DoLP',
    aolp_dir= root_dir + '/synthetic/bun-zipper-front/params/AoLP',
    synthesis_normals_dir= root_dir + '/synthetic/bun-zipper-front/synthesis-normals',
                                                                           mask_dir= root_dir + '/synthetic/bun-zipper-front/masks',
                                                                           label_dir= root_dir + '/synthetic/bun-zipper-front/normals-png',
                                                                           transform=augs_train)
dataset_synthetic_polar_armadillo_back = TransSfPDataset(
    dolp_dir= root_dir + '/synthetic/armadillo-back/params/DoLP',
    aolp_dir= root_dir + '/synthetic/armadillo-back/params/AoLP',
    synthesis_normals_dir= root_dir + '/synthetic/armadillo-back/synthesis-normals',
                                                                         mask_dir= root_dir + '/synthetic/armadillo-back/masks',
                                                                         label_dir= root_dir + '/synthetic/armadillo-back/normals-png',
                                                                         transform=augs_train)
dataset_synthetic_polar_armadillo_front = TransSfPDataset(
    dolp_dir= root_dir + '/synthetic/armadillo-front/params/DoLP',
    aolp_dir= root_dir + '/synthetic/armadillo-front/params/AoLP',
    synthesis_normals_dir= root_dir + '/synthetic/armadillo-front/synthesis-normals',
                                                                          mask_dir= root_dir + '/synthetic/armadillo-front/masks',
                                                                          label_dir= root_dir + '/synthetic/armadillo-front/normals-png',
                                                                          transform=augs_train)
dataset_synthetic_polar_dragon_vrip = TransSfPDataset(
    dolp_dir= root_dir + '/synthetic/dragon-vrip/params/DoLP',
    aolp_dir= root_dir + '/synthetic/dragon-vrip/params/AoLP',
    synthesis_normals_dir= root_dir + '/synthetic/dragon-vrip/synthesis-normals',
                                                                      mask_dir= root_dir + '/synthetic/dragon-vrip/masks',
                                                                      label_dir= root_dir + '/synthetic/dragon-vrip/normals-png',
                                                                      transform=augs_train)
dataset_synthetic_polar_happy_vrip_back= TransSfPDataset(
    dolp_dir= root_dir + '/synthetic/happy-vrip-back/params/DoLP',
    aolp_dir= root_dir + '/synthetic/happy-vrip-back/params/AoLP',
    synthesis_normals_dir= root_dir + '/synthetic/happy-vrip-back/synthesis-normals',
                                                                         mask_dir= root_dir + '/synthetic/happy-vrip-back/masks',
                                                                         label_dir= root_dir + '/synthetic/happy-vrip-back/normals-png',
                                                                         transform=augs_train)
dataset_synthetic_polar_happy_vrip_front= TransSfPDataset(
    dolp_dir= root_dir + '/synthetic/happy-vrip-front/params/DoLP',
    aolp_dir= root_dir + '/synthetic/happy-vrip-front/params/AoLP',
    synthesis_normals_dir= root_dir + '/synthetic/happy-vrip-front/synthesis-normals',
                                                                          mask_dir= root_dir + '/synthetic/happy-vrip-front/masks',
                                                                          label_dir= root_dir + '/synthetic/happy-vrip-front/normals-png',
                                                                          transform=augs_train)
dataset_synthetic_polar_middle_round_cup = TransSfPDataset(
    dolp_dir= root_dir + '/synthetic/middle-round-cup/params/DoLP',
    aolp_dir= root_dir + '/synthetic/middle-round-cup/params/AoLP',
    synthesis_normals_dir= root_dir + '/synthetic/middle-round-cup/synthesis-normals',
                                                                           mask_dir= root_dir + '/synthetic/middle-round-cup/masks',
                                                                           label_dir= root_dir + '/synthetic/middle-round-cup/normals-png',
                                                                           transform=augs_train)
dataset_synthetic_polar_bear_front = TransSfPDataset(
    dolp_dir= root_dir + '/synthetic/bear-front/params/DoLP',
    aolp_dir= root_dir + '/synthetic/bear-front/params/AoLP',
    synthesis_normals_dir= root_dir + '/synthetic/bear-front/synthesis-normals',
                                                                     mask_dir= root_dir + '/synthetic/bear-front/masks',
                                                                     label_dir= root_dir + '/synthetic/bear-front/normals-png',
                                                                     transform=augs_train)
dataset_synthetic_polar_cow_front = TransSfPDataset(
    dolp_dir= root_dir + '/synthetic/cow-front/params/DoLP',
    aolp_dir= root_dir + '/synthetic/cow-front/params/AoLP',
    synthesis_normals_dir= root_dir + '/synthetic/cow-front/synthesis-normals',
                                                                    mask_dir= root_dir + '/synthetic/cow-front/masks',
                                                                    label_dir= root_dir + '/synthetic/cow-front/normals-png',
                                                                    transform=augs_train)
dataset_synthetic_polar_cow_back = TransSfPDataset(
    dolp_dir= root_dir + '/synthetic/cow-back/params/DoLP',
    aolp_dir= root_dir + '/synthetic/cow-back/params/AoLP',
    synthesis_normals_dir= root_dir + '/synthetic/cow-back/synthesis-normals',
                                                                   mask_dir= root_dir + '/synthetic/cow-back/masks',
                                                                   label_dir= root_dir + '/synthetic/cow-back/normals-png',
                                                                   transform=augs_train)
dataset_synthetic_polar_pot_back = TransSfPDataset(
    dolp_dir= root_dir + '/synthetic/pot-back/params/DoLP',
    aolp_dir= root_dir + '/synthetic/pot-back/params/AoLP',
    synthesis_normals_dir= root_dir + '/synthetic/pot-back/synthesis-normals',
                                                                   mask_dir= root_dir + '/synthetic/pot-back/masks',
                                                                   label_dir= root_dir + '/synthetic/pot-back/normals-png',
                                                                   transform=augs_train)
dataset_synthetic_polar_pot_front = TransSfPDataset(
    dolp_dir= root_dir + '/synthetic/pot-front/params/DoLP',
    aolp_dir= root_dir + '/synthetic/pot-front/params/AoLP',
    synthesis_normals_dir= root_dir + '/synthetic/pot-front/synthesis-normals',
                                                                   mask_dir= root_dir + '/synthetic/pot-front/masks',
                                                                   label_dir= root_dir + '/synthetic/pot-front/normals-png',
                                                                   transform=augs_train)



db_list_synthetic  = [dataset_synthetic_polar_bun_zipper_back,dataset_synthetic_polar_bun_zipper_front,dataset_synthetic_polar_armadillo_back,dataset_synthetic_polar_armadillo_front,
                      dataset_synthetic_polar_dragon_vrip,dataset_synthetic_polar_happy_vrip_back,dataset_synthetic_polar_happy_vrip_front,
                      dataset_synthetic_polar_middle_round_cup,dataset_synthetic_polar_bear_front,dataset_synthetic_polar_cow_front,dataset_synthetic_polar_cow_back,
                      dataset_synthetic_polar_pot_back,dataset_synthetic_polar_pot_front]
db_list_real = [dataset_middle_square_cup_black_background_12_28,dataset_middle_round_cup_black_background_12_28,dataset_middle_white_cup_black_background_12_28]
db_train_list = db_list_synthetic + db_list_real


dataset = torch.utils.data.ConcatDataset(db_train_list)

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
                         pin_memory=pin_memory)
testLoader_tiny_white_cup = DataLoader(dataset_tiny_white_cup_black_background_12_28,
                         batch_size=1,
                         num_workers=num_workers,
                         drop_last=True,
                         pin_memory=pin_memory)
testLoader_tiny_white_cup_edges = DataLoader(dataset_tiny_white_cup_edges_black_background_12_28,
                         batch_size=1,
                         num_workers=num_workers,
                         drop_last=True,
                         pin_memory=pin_memory)
testLoader_bird_back = DataLoader(dataset_bird_back_1_20,
                         batch_size=1,
                         num_workers=num_workers,
                         drop_last=True,
                         pin_memory=pin_memory)
testLoader_bird_front = DataLoader(dataset_bird_front_1_20,
                         batch_size=1,
                         num_workers=num_workers,
                         drop_last=True,
                         pin_memory=pin_memory)
testLoader_cat_back = DataLoader(dataset_cat_back_1_20,
                         batch_size=1,
                         num_workers=num_workers,
                         drop_last=True,
                         pin_memory=pin_memory)
testLoader_cat_front = DataLoader(dataset_cat_front_1_20,
                         batch_size=1,
                         num_workers=num_workers,
                         drop_last=True,
                         pin_memory=pin_memory)
testLoader_hemi_sphere_big = DataLoader(dataset_hemi_sphere_big_1_20,
                         batch_size=1,
                         num_workers=num_workers,
                         drop_last=True,
                         pin_memory=pin_memory)
testLoader_hemi_sphere_small = DataLoader(dataset_hemi_sphere_small_1_20,
                         batch_size=1,
                         num_workers=num_workers,
                         drop_last=True,
                         pin_memory=pin_memory)



print("trainLoader size:",trainLoader.__len__()*trainLoader.batch_size)

###################### ModelBuilder #############################

#-- 1、 config parameters
backbone_model = 'resnet50'
use_atten = False

#-- 2、create model
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TranSfPNet(backbone = backbone_model,device=device)
#-- 3、Enable GPU for training

model = model.to(device)


###################### Setup Optimizer #############################


#-- 1、 config parameters
learningRate = 1e-6
weightDecay = 5e-4
momentum = 0.9
# lrSchedulerStep
lrScheduler = 'StepLR'
step_size = 9
gamma = 0.1


#-- 2、create optimizer
optimizer = torch.optim.Adam(model.parameters(),
                            lr=float(learningRate),
                            weight_decay=float(weightDecay))

#-- 3、create learningRate schduler
if not lrScheduler:
    pass
elif lrScheduler == 'StepLR':
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=step_size,
                                                   gamma=float(gamma))
else:
    raise ValueError(
        "Invalid Scheduler from config file: '{}'. Valid values are ['', 'StepLR', 'ReduceLROnPlateau']".format(
            lrScheduler))

#-- 4、select koss fu
criterion = loss_functions.my_loss_cosine
writer = SummaryWriter()


###################### Train Model #############################
#-- 1、config parameters
MAX_EPOCH = 20
total_iter_num = 0
START_EPOCH = 0

#-- 2、epoch cycle
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
        aolp = params_t[:,1,:,:]
        normals_t = normals_t.to(device)
        label_t = label_t.to(device)
        # Forward + Backward Prop
        start = time.time()
        optimizer.zero_grad()
        torch.set_grad_enabled(True)
        with torch.autograd.set_detect_anomaly(True):
            normal_vectors,atten_map = model(params_t,normals_t)
            normal_vectors_norm = nn.functional.normalize(normal_vectors.double(), p=2, dim=1)
            normal_vectors_norm = normal_vectors_norm
            loss = criterion(normal_vectors_norm, label_t.double(),mask_tensor=mask_t,atten_map = atten_map,aolp = aolp,reduction='elementwise_mean',device=device,use_atten = use_atten)
        loss /= batch_size
        loss.backward()
        optimizer.step()

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
                os.path.join(code_root_dir + '/results/train', str(iter_num).zfill(3) + '-label.png'),
                label_t_rgb)
            API.utils.png_saver(
                os.path.join(code_root_dir + '/results/train', str(iter_num).zfill(3) + '-predict.png'),
                predict_norm_rgb)
            API.utils.png_saver(
                os.path.join(code_root_dir + '/results/train', str(iter_num).zfill(3) + '-atten.png'),
                atten_map_rgb)


    num_samples = (len(trainLoader))
    epoch_loss = running_loss/num_samples
    print("train running loss:",epoch_loss)
    print("train running mean:",running_mean/num_samples)
    print("train running median:",running_median/num_samples)



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
            testLoader= testLoader_tiny_white_cup_edges,device=device,criterion=criterion,use_atten=use_atten,epoch = epoch,name = 'tiny_white_cup_edges',writer=writer,resultPath= code_root_dir + '/results/tiny-white-cup-edges')
    print('tiny-white-cup-edges:\n')
    print('loss: ',running_loss)
    print('mean: ',running_mean)
    print('median: ',running_median)
    print('percentage_1: ',running_percentage_1)
    print('percentage_2: ',running_percentage_2)
    print('percentage_3: ',running_percentage_3)
    print('=' * 10)
    print('\n')
    mean_all +=running_mean
    median_all += running_median
    acc_all_1 +=running_percentage_1
    acc_all_2 +=running_percentage_2
    acc_all_3 +=running_percentage_3

    count +=1

    running_loss,running_mean,running_median,running_percentage_1,running_percentage_2,running_percentage_3 = evaluation(model = model,
            testLoader= testLoader_tiny_white_cup,device=device,criterion=criterion,use_atten=use_atten,epoch = epoch,name = 'tiny_white_cup',writer=writer,resultPath= code_root_dir + '/results/tiny-white-cup')
    print('tiny-white-cup:')
    print('loss: ',running_loss)
    print('mean: ',running_mean)
    print('median: ',running_median)
    print('percentage_1: ',running_percentage_1)
    print('percentage_2: ',running_percentage_2)
    print('percentage_3: ',running_percentage_3)
    print('=' * 10)
    print('\n')
    mean_all +=running_mean
    median_all += running_median
    acc_all_1 +=running_percentage_1
    acc_all_2 +=running_percentage_2
    acc_all_3 +=running_percentage_3
    count += 1

    running_loss,running_mean,running_median,running_percentage_1,running_percentage_2,running_percentage_3 = evaluation(model = model,
            testLoader= testLoader_bird_front,device=device,criterion=criterion,use_atten=use_atten,epoch = epoch,name = 'bird_front',writer=writer,resultPath= code_root_dir + '/results/bird-front')
    print('bird-front:')
    print('loss: ',running_loss)
    print('mean: ',running_mean)
    print('median: ',running_median)
    print('percentage_1: ',running_percentage_1)
    print('percentage_2: ',running_percentage_2)
    print('percentage_3: ',running_percentage_3)
    acc_all_1 +=running_percentage_1
    acc_all_2 +=running_percentage_2
    acc_all_3 +=running_percentage_3
    print('=' * 10)
    print('\n')

    running_loss,running_mean,running_median,running_percentage_1,running_percentage_2,running_percentage_3 = evaluation(model = model,
            testLoader= testLoader_bird_back,device=device,criterion=criterion,use_atten=use_atten,epoch = epoch,name = 'bird_back',writer=writer,resultPath= code_root_dir + '/results/bird-back')
    print('bird-back:')
    print('loss: ',running_loss)
    print('mean: ',running_mean)
    print('median: ',running_median)
    print('percentage_1: ',running_percentage_1)
    print('percentage_2: ',running_percentage_2)
    print('percentage_3: ',running_percentage_3)
    acc_all_1 +=running_percentage_1
    acc_all_2 +=running_percentage_2
    acc_all_3 +=running_percentage_3
    print('=' * 10)
    print('\n')
    mean_all +=running_mean
    median_all += running_median
    count +=1

    running_loss,running_mean,running_median,running_percentage_1,running_percentage_2,running_percentage_3 = evaluation(model = model,
            testLoader= testLoader_cat_front,device=device,criterion=criterion,use_atten=use_atten,epoch = epoch,name = 'cat_front',writer=writer,resultPath= code_root_dir + '/results/cat-front')
    print('cat-front:')
    print('loss: ',running_loss)
    print('mean: ',running_mean)
    print('median: ',running_median)
    print('percentage_1: ',running_percentage_1)
    print('percentage_2: ',running_percentage_2)
    print('percentage_3: ',running_percentage_3)
    acc_all_1 +=running_percentage_1
    acc_all_2 +=running_percentage_2
    acc_all_3 +=running_percentage_3
    print('=' * 10)
    print('\n')
    mean_all +=running_mean
    median_all += running_median
    count +=1

    running_loss,running_mean,running_median,running_percentage_1,running_percentage_2,running_percentage_3 = evaluation(model = model,
            testLoader= testLoader_cat_back,device=device,criterion=criterion,use_atten=use_atten,epoch = epoch,name = 'cat_back',writer=writer,resultPath= code_root_dir + '/results/cat-back')
    print('cat-back:')
    print('loss: ',running_loss)
    print('mean: ',running_mean)
    print('median: ',running_median)
    print('percentage_1: ',running_percentage_1)
    print('percentage_2: ',running_percentage_2)
    print('percentage_3: ',running_percentage_3)
    acc_all_1 +=running_percentage_1
    acc_all_2 +=running_percentage_2
    acc_all_3 +=running_percentage_3
    print('=' * 10)
    print('\n')
    mean_all +=running_mean
    median_all += running_median
    count +=1

    running_loss,running_mean,running_median,running_percentage_1,running_percentage_2,running_percentage_3 = evaluation(model = model,
            testLoader= testLoader_hemi_sphere_big,device=device,criterion=criterion,use_atten=use_atten,epoch = epoch,name = 'hemi_sphere_big',writer=writer,resultPath= code_root_dir + '/results/hemi-sphere-big')
    print('hemi-sphere-big:')
    print('loss: ',running_loss)
    print('mean: ',running_mean)
    print('median: ',running_median)
    print('percentage_1: ',running_percentage_1)
    print('percentage_2: ',running_percentage_2)
    print('percentage_3: ',running_percentage_3)
    acc_all_1 +=running_percentage_1
    acc_all_2 +=running_percentage_2
    acc_all_3 +=running_percentage_3
    print('=' * 10)
    print('\n')
    mean_all +=running_mean
    median_all += running_median
    count +=1

    running_loss,running_mean,running_median,running_percentage_1,running_percentage_2,running_percentage_3 = evaluation(model = model,
            testLoader= testLoader_hemi_sphere_small,device=device,criterion=criterion,use_atten=use_atten,epoch = epoch,name = 'hemi_sphere_small',writer=writer,resultPath= code_root_dir + '/results/hemi-sphere-small')
    print('hemi-sphere-small:')
    print('loss: ',running_loss)
    print('mean: ',running_mean)
    print('median: ',running_median)
    print('percentage_1: ',running_percentage_1)
    print('percentage_2: ',running_percentage_2)
    print('percentage_3: ',running_percentage_3)
    acc_all_1 +=running_percentage_1
    acc_all_2 +=running_percentage_2
    acc_all_3 +=running_percentage_3
    print('=' * 10)
    print('\n')
    mean_all +=running_mean
    median_all += running_median
    count +=1

    print('all mean: ',mean_all/count)
    print('all median: ',median_all/count)
    print('percentage 1: ',acc_all_1/count)
    print('percentage 2: ',acc_all_2/count)
    print('percentage 3: ',acc_all_3/count)
    mean_list.append(mean_all/count)
    median_list.append(median_all/count)

