import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
import modeling.backbone.resnet
class NetLab(nn.Module):
    def __init__(self,network = 'resnet18',num_classes = 10):
        super(NetLab, self).__init__()
        if(network == 'resnet18'):
            self.model = resnet18(num_classes)
        if(network == 'resnet50'):
            self.model = resnet50(num_classes)
        if(network == 'myCNN'):
            self.model = myCNN(num_classes)
        if(network=='mobilenet'):
            self.model = mobilenet(num_classes)
        if(network=='myNN'):
            self.model = myNN(num_classes)
    def forward(self, x):
        x = self.model(x)
        return x

class myNN(nn.Module):
    def __init__(self,num_classes = 10):
        super(myNN, self).__init__()
        self.linear1 = nn.Linear(28*28,256)
        self.linear2 = nn.Linear(256,512)
        self.linear3 = nn.Linear(512,128)
        self.linear4 = nn.Linear(128,10)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = x.reshape(-1,28*28)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.linear4(x)
        return x

class myCNN(nn.Module):
    def __init__(self,num_classed = 10):
        super(myCNN,self).__init__()
        self.conv1 = nn.Conv2d(1,16,kernel_size=3,stride = 2,padding=3)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride = 2,padding=1)
        self.conv2 = nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.linear1 = nn.Linear(4*4*32,25)
        self.linear2 = nn.Linear(256,10)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(-1, 4*4*32)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class resnet18(nn.Module):
    def __init__(self,num_classes=10):
        super(resnet18, self).__init__()
        self.res18 = models.resnet18(pretrained=True)

        numFit = self.res18.fc.in_features
        self.res18.fc = nn.Linear(numFit, num_classes)

    def forward(self, x):
        x = self.res18(x)
        return x

class resnet50(nn.Module):
    def __init__(self,num_classes=10):
        super(resnet50, self).__init__()
        self.res50 = models.resnet50(pretrained=True)

        numFit = self.res50.fc.in_features
        self.res50.fc = nn.Linear(numFit, num_classes)

    def forward(self, x):
        x = self.res50(x)
        return x
class mobilenet(nn.Module):
    def __init__(self,num_classes = 10):
        super(mobilenet, self).__init__()
        self.mobilenet = models.mobilenet_v2(num_classes = num_classes,pretrained=False)
    def forward(self, x):
        x = self.mobilenet(x)
        return x


