import torch
import torch.nn as nn
import math

from torch.nn.init import kaiming_normal_
import modeling.PS_FCN.model_utils as model_utils
import modeling.utils.SKBlock as SKBlock
import modeling.utils.ESPABlock as ESPABlock
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.smqNet.decoder  import build_decoder
from modeling.backbone import build_backbone
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
import matplotlib.pyplot as pt
from modeling.deepSfp.SPADE import SPADE
from modeling.attention.DANet import DAModule
import numpy as np
class FinalLayer(nn.Module):
    def __init__(self,input_channel,output_channel):
        super(FinalLayer, self).__init__()
        ks = 3
        pw = ks//2
        self.conv1 = nn.Conv2d(in_channels=input_channel,out_channels=input_channel,kernel_size=ks,padding=pw)
        self.instace_norm = nn.InstanceNorm2d(num_features=input_channel)
        self.leaky_relu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(in_channels=input_channel,out_channels=output_channel,kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.instace_norm(x1)
        x1 = self.leaky_relu(x1)
        x = x1 + x
        x = self.conv2(x)
        return x

class calibrator(nn.Module):
    def __init__(self,in_channels,out_channels,output_size,atten = False):
        super(calibrator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels,out_channels = in_channels,kernel_size=3,padding=3//2)
        self.leaky_relu = nn.LeakyReLU()
        self.bn1 = nn.InstanceNorm2d(num_features=in_channels)
        self.atten = atten
        if(atten):
            self.attention = DAModule(d_model=in_channels,kernel_size=3,H = output_size,W = output_size)

        self.conv2 = nn.Conv2d(in_channels = in_channels,out_channels=out_channels,kernel_size=1)
        self.bn2 = nn.InstanceNorm2d(num_features=out_channels)
        self.size = output_size
    def forward(self,x):
        x = F.interpolate(x,size=(self.size,self.size),mode='bilinear',align_corners=False)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)

        if(self.atten):
            x = self.attention(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        return x
class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
    def forward(self,q,k,v):
        '''
        Computes
        :param q: Queries (B, C, H, W)
        :param k: Keys (B, C, H, W)
        :param v: Values (B, C, H, W)
        :return:
        '''
        B,C,H,W = q.shape
        H = int(H/2)
        W = int(W/2)
        q = F.interpolate(q,size=(H,W),mode='bilinear',align_corners=False)
        k = F.interpolate(k,size=(H,W),mode='bilinear',align_corners=False)
        v = F.interpolate(v,size=(H,W),mode='bilinear',align_corners=False)

        q = q.view(B,1,C,H,W).view(B,1,C,H*W).permute(0,1,3,2) # (B,h,HxW,C)
        k = k.view(B,1,C,H,W).view(B,1,C,H*W) # (B,h,C,HxW)
        v = v.view(B,1,C,H,W).view(B,1,C,H*W).permute(0,1,3,2) # (B,h,HxW,C)
        attn = torch.matmul(q, k) / np.sqrt(H*W) # (B,h,C,HxW)
        attn = torch.softmax(attn,dim = -1)
        out = torch.matmul(attn, v).permute(0, 1, 3, 2) # (B,h,C,HxW)
        out = out.view(B,C,H,W)
        H*=2
        W*=2
        out = F.interpolate(out,size=(H,W),mode='bilinear',align_corners=False)


        return out

class upSample(nn.Module):
    def __init__(self,in_channels,out_channels,output_size):
        super(upSample, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels*2,out_channels = in_channels,kernel_size=3,padding=3//2)
        self.leaky_relu = nn.LeakyReLU()
        self.bn1 = nn.InstanceNorm2d(num_features=in_channels)
        self.conv2 = nn.Conv2d(in_channels = in_channels,out_channels=out_channels,kernel_size=1)
        self.bn2 = nn.InstanceNorm2d(num_features=out_channels)
        self.size = output_size
    def forward(self,x,skip):
        x = torch.cat((x,skip),dim = 1)
        x = F.interpolate(x,size=(self.size,self.size),mode='bilinear',align_corners=True)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)

        return x

class decoder(nn.Module):
    def __init__(self,out_channel):
        super(decoder, self).__init__()
        self.up4 = upSample(in_channels=256,out_channels=128,output_size=64)
        self.up3 = upSample(in_channels=128,out_channels=64,output_size=128)
        self.up2 = upSample(in_channels=64,out_channels=32,output_size=256)
        self.up1 = upSample(in_channels=32,out_channels=16,output_size=512)
        self.finalLayer = FinalLayer(input_channel=16*2,output_channel=out_channel)

    def forward(self,x,x_0,x_1,x_2,x_3,x_4):
        # x_0: (B,16,512,512)
        # x_1: (B,32,256,256)
        # x_2: (B,64,128,128)
        # x_3: (B,128,64,64)
        # x_4: (B,256,32,32)

        x = self.up4(x,x_4) #(B,128,64,64)
        x = self.up3(x,x_3) #(B,64,128,128)
        x = self.up2(x,x_2) #(B,32,256,256)
        x = self.up1(x,x_1) #B(B,16,512,512)

        x = torch.cat((x,x_0),dim = 1)
        x = self.finalLayer(x) #(B,3,512,512)
        return x



class TransRemove(nn.Module):
    def __init__(self, backbone='resnet50', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False,device= None):
        super(TransRemove, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d



        # branch
        self.backbone_orig = build_backbone(in_channels=4 ,backbone=backbone, output_stride=output_stride, BatchNorm=BatchNorm,Fusion=True)
        self.aspp_orig = build_aspp(backbone, output_stride, BatchNorm)


        self.decoder = decoder(out_channel=2)

        self.calibrator_0 = calibrator(in_channels=64,out_channels=16,output_size=512)
        self.calibrator_1 = calibrator(in_channels=64,out_channels=32,output_size=256)
        self.calibrator_2 = calibrator(in_channels=256,out_channels=64,output_size=128)
        self.calibrator_3 = calibrator(in_channels=512,out_channels=128,output_size=64)
        self.calibrator_4 = calibrator(in_channels=1024,out_channels=256,output_size=32)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        if freeze_bn:
            self.freeze_bn()

        self.attention_layer = Attention()




        self.writer = SummaryWriter()

    def forward(self, params):




        # encoder
        x,x_0,x_1,x_2,x_3,x_4 = self.backbone_orig(params)
        x = self.aspp_orig(x)


        x_0 = self.calibrator_0(x_0)
        x_1 = self.calibrator_1(x_1)
        x_2 = self.calibrator_2(x_2)
        x_3 = self.calibrator_3(x_3)
        x_4 = self.calibrator_4(x_4)

        # decode
        x = self.decoder(x,x_0,x_1,x_2,x_3,x_4)

        x = self.sigmoid(x)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()
if __name__ == "__main__":
    input = torch.ones(6,2,512,512)
    model = TransRemove(num_classes=2)
    print(model(input))