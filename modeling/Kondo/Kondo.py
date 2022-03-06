""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class conv1(nn.Module):

    def __init__(self, c_in, c_out):
        super(conv1,self).__init__()
        self.conv = nn.Conv2d(in_channels=c_in,out_channels=c_out,kernel_size=3,stride=1,padding=1)
        self.bn = nn.BatchNorm2d(num_features=c_out)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class conv2(nn.Module):

    def __init__(self, c_in,c_out):
        super(conv2,self).__init__()
        self.conv = nn.Conv2d(in_channels=c_in,out_channels=c_out,kernel_size=4,stride=2)
    def forward(self, x):
        x = self.conv(x)
        return x

class Dconv(nn.Module):

    def __init__(self, c_in, c_out):
        super(Dconv,self).__init__()
        self.conv = nn.ConvTranspose2d(c_in, c_out, kernel_size=3, stride=2)
    def forward(self, x):
        return self.conv(x)
class conv3(nn.Module):
    def __init__(self, c_in, c_out):
        super(conv3,self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3)

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    def __init__(self, c_in,c_out,isLast = False):
        super(Down,self).__init__()
        self.conv0 = conv1(c_in = c_in,c_out = c_out)
        self.conv1= conv1(c_in = c_out,c_out = c_out)
        self.conv2= conv1(c_in = c_out,c_out = c_out)
        self.conv3= conv2(c_in = c_out,c_out = c_out)
        self.isLast = isLast
        if(self.isLast):
            self.conv_last =conv1(c_in=c_out,c_out=c_out)
    def forward(self, x):

        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        if(self.isLast):
            x = self.conv_last(x)
        x = self.conv3(x)
        return x
class BottleNeck(nn.Module):
    def __init__(self, c_in,c_out):
        super(BottleNeck,self).__init__()
        self.conv0 = conv1(c_in = c_in,c_out = 512)
        self.conv1= conv1(c_in = 512,c_out = 512)
        self.conv2= conv1(c_in = 512,c_out = 512)
        self.conv3 = conv2(c_in = 512,c_out=512)
        self.conv4 = conv1(c_in = 512,c_out =512)
        self.conv5 = conv1(c_in=512,c_out=c_out)
        self.conv6 = conv1(c_in=c_out,c_out=c_out)
        self.conv7 = conv1(c_in=c_out,c_out=c_out)
    def forward(self, x):

        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        return x
class Up(nn.Module):
    def __init__(self, c_in,c_out,isLast = True):
        super(Up,self).__init__()
        self.conv0 = Dconv(c_in=c_in,c_out=c_in)
        self.conv1 = conv1(c_in=c_in,c_out=c_out)
        self.conv2 = conv1(c_in=c_out,c_out=c_out)


    def forward(self, x,skip):
        x = self.conv0(x)
        x = F.interpolate(x, size=(skip.shape[2], skip.shape[3]), mode='bilinear', align_corners=True)
        x = x+skip
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Kondo(nn.Module):
    def __init__(self, c_in,device):
        super(Kondo,self).__init__()
        self.Down0 = Down(c_in = c_in,c_out = 64)
        self.Down1 = Down(c_in = 64,c_out = 128)
        self.Down2= Down(c_in = 128,c_out = 256,isLast=True)
        self.bottle = BottleNeck(c_in=256,c_out=256)
        self.Up2 = Up(c_in = 256,c_out = 128)
        self.Up1 = Up(c_in=128,c_out=64)
        self.Up0 = Up(c_in=64,c_out=64)
        self.output = conv3(c_in=64,c_out=3)


        self.kernel_size = 9
        self.lamda = 1
        self.m = 0.5
        self.mean_kernel = torch.ones([1,1,self.kernel_size,self.kernel_size])/self.kernel_size**2
        self.mean_kernel = self.mean_kernel.to(device)
        self.mean_kernel = nn.Parameter(data=self.mean_kernel, requires_grad=False)
        self.sum_kernel_1 = torch.ones([1,1,self.kernel_size,self.kernel_size])
        self.sum_kernel_1 = self.sum_kernel_1.to(device)
        self.sum_kernel_1 = nn.Parameter(data=self.sum_kernel_1, requires_grad=False)

        self.sum_kernel_3 = torch.ones([3,3,self.kernel_size,self.kernel_size])
        self.sum_kernel_3 = self.sum_kernel_3.to(device)
        self.sum_kernel_3 = nn.Parameter(data=self.sum_kernel_3, requires_grad=False)

    def forward(self, params,normals):
        img  = params
        img_split = torch.split(img, 1, 1)
        aolp = img_split[1]

        # get attention map
        mean_map = nn.functional.conv2d(aolp,self.mean_kernel,padding=self.kernel_size//2)
        abs_map =torch.abs(aolp - mean_map)
        abs_map = torch.pow(abs_map,self.m)
        atten_map = nn.functional.conv2d(abs_map,self.sum_kernel_1,padding=self.kernel_size//2)
        shape = atten_map.shape
        atten_map = torch.reshape(atten_map,[shape[0],-1])
        max_values,indices = torch.max(atten_map,dim = 1)
        max_values = torch.reshape(max_values,[shape[0],1])
        atten_map = torch.div(atten_map,max_values)
        atten_map = torch.reshape(atten_map,[shape[0],shape[1],shape[2],shape[3]])

        # x = torch.cat((params,normals,atten_map),dim=1)
        x = params
        x0 = self.Down0(x)
        x1 = self.Down1(x0)
        x2 = self.Down2(x1)
        x = self.bottle(x2)

        x = self.Up2(x,x2)
        x = self.Up1(x,x1)
        x = self.Up0(x,x0)
        x = self.output(x)

        x = F.interpolate(x, size=(shape[2], shape[3]), mode='bilinear', align_corners=True)

        return x,x


if __name__ =='__main__':
    x = torch.ones(6,13,512,512)
    model = Kondo(c_in=13)
    y = model(x)
    print(y.shape)
