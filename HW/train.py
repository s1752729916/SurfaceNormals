from termcolor import colored
import torch
import torch.nn as nn
from imgaug import augmenters as iaa
from torch.utils.data import DataLoader,SubsetRandomSampler
from torchvision import transforms
from tqdm import tqdm
from modeling import deeplab
from Net import *
from mydataloader import *

#-- 0、params
validation_split = 0.2
batch_size = 32
num_workers = 2
shuffle_dataset = True
network = 'mobilenet'
learningRate = 1e-6
weightDecay = 5e-4
MAX_EPOCH = 100

#-- 1、Augmentation
augs_train = iaa.Sequential([
    # Geometric Augs
    # iaa.Resize({"height": imgHeight, "width": imgWidth }, interpolation='nearest'),  # Resize image
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Rot90((0, 4)),
    # Blur and Noise
    iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 1.5), name="gaus-blur")),
    iaa.Sometimes(0.1, iaa.Grayscale(alpha=(0.0, 1.0), from_colorspace="RGB", name="grayscale")),
    iaa.Sometimes(0.2, iaa.AdditiveLaplaceNoise(scale=(0, 0.1*255), per_channel=True, name="gaus-noise")),

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

#-- 2、Create Dataset
dog_dataset = AnimalFruitsDataset(img_dir='/home/zjw/smq/SurfaceNormals/HW/problem1/Data/动物/动物/狗',label_number = 0,transform=augs_train,input_only = input_only)
horse_dataset = AnimalFruitsDataset(img_dir='/home/zjw/smq/SurfaceNormals/HW/problem1/Data/动物/动物/马',label_number = 1,transform=augs_train,input_only = input_only)
cat_dataset = AnimalFruitsDataset(img_dir='/home/zjw/smq/SurfaceNormals/HW/problem1/Data/动物/动物/猫',label_number = 2,transform=augs_train,input_only = input_only)
cow_dataset = AnimalFruitsDataset(img_dir='/home/zjw/smq/SurfaceNormals/HW/problem1/Data/动物/动物/牛',label_number = 3,transform=augs_train,input_only = input_only)
pig_dataset = AnimalFruitsDataset(img_dir='/home/zjw/smq/SurfaceNormals/HW/problem1/Data/动物/动物/猪',label_number = 4,transform=augs_train,input_only = input_only)

orange_dataset = AnimalFruitsDataset(img_dir='/home/zjw/smq/SurfaceNormals/HW/problem1/Data/水果/水果/橙子',label_number = 5,transform=augs_train,input_only = input_only)
durian_dataset = AnimalFruitsDataset(img_dir='/home/zjw/smq/SurfaceNormals/HW/problem1/Data/水果/水果/榴莲',label_number = 6,transform=augs_train,input_only = input_only)
apple_dataset = AnimalFruitsDataset(img_dir='/home/zjw/smq/SurfaceNormals/HW/problem1/Data/水果/水果/苹果',label_number = 7,transform=augs_train,input_only = input_only)
grape_dataset = AnimalFruitsDataset(img_dir='/home/zjw/smq/SurfaceNormals/HW/problem1/Data/水果/水果/葡萄',label_number = 8,transform=augs_train,input_only = input_only)
banana_dataset = AnimalFruitsDataset(img_dir='/home/zjw/smq/SurfaceNormals/HW/problem1/Data/水果/水果/香蕉',label_number = 9,transform=augs_train,input_only = input_only)
db_list = [dog_dataset,horse_dataset,cat_dataset,cow_dataset,pig_dataset,orange_dataset,durian_dataset,apple_dataset,grape_dataset,banana_dataset]
dataset = torch.utils.data.ConcatDataset(db_list)

#-- 3、create dataloader
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
                         batch_size=batch_size,
                         num_workers=num_workers,
                         drop_last=True,
                         sampler=train_sampler)
testLoader = DataLoader(dataset,
                         batch_size=1,
                         num_workers=num_workers,
                         drop_last=True,
                         sampler=valid_sampler)
print("trainLoader size:",trainLoader.__len__()*trainLoader.batch_size)
print("testLoader size:",testLoader.__len__()*testLoader.batch_size)

#-- 4、Create Model
model = NetLab(network= network,num_classes=10)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
model = model.to(device)

#-- 5、Create Optimizer
optimizer = torch.optim.Adam(model.parameters(),
                            lr=float(learningRate),
                            weight_decay=float(weightDecay))
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

#-- 6、Train Cycle
for epoch in range(0,MAX_EPOCH):
    print('\n\nEpoch {}/{}'.format(epoch, MAX_EPOCH - 1))
    print('-' * 30)
    ###################### Training Cycle #############################
    print('Train:')
    print('=' * 10)
    model.train()  # set model mode to train mode
    running_loss = 0.0
    for iter_num, batch in enumerate(tqdm(trainLoader)):
        inputs_t, label = batch
        inputs_t = inputs_t.to(device)
        # print('input_t',inputs_t.detach().shape)
        label = label.to(device)
        optimizer.zero_grad()
        torch.set_grad_enabled(True)
        outputs = model(inputs_t)
        loss = loss_fn(outputs,label)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
    print("train loss",running_loss/trainLoader.__len__()/batch_size)

    ###################### Test Cycle #############################
    model.eval()
    accuracy = 0
    running_loss = 0
    for item_num,batch in enumerate(tqdm(testLoader)):
        inputs_t,label = batch
        inputs_t = inputs_t.to(device)
        # print('input_t',inputs_t.detach().shape)
        label = label.to(device)
        with torch.no_grad():
            outputs = model(inputs_t)
        running_loss = loss_fn(outputs,label)
        accuracy = accuracy + (outputs.argmax(1) == label).sum()  # 正确率的内容
    print('test loss',running_loss/len(testLoader))
    print('accuracy',accuracy/len(testLoader)*100)




