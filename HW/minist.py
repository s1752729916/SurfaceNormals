import torchvision.datasets
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
network = 'myCNN'
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


#-- 3、create dataloader
trainLoader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/',train=True,download=True,transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,),(0.3081))
    ])),batch_size=batch_size,shuffle=shuffle_dataset,drop_last=True)
testLoader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/',train=False,download=True,transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,),(0.3081))
    ])),batch_size = batch_size,shuffle = shuffle_dataset)



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
        running_loss += loss_fn(outputs,label)
        accuracy = accuracy + (outputs.argmax(1) == label).sum()  # 正确率的内容

    print('test loss',running_loss/len(testLoader)/batch_size)
    print('accuracy',accuracy/len(testLoader)/batch_size*100)




