'''
implemetaion of <deep shape from polarization,ECCV,2020>
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.deepSfp.feature_extractor import FeatureExtractor
from modeling.deepSfp.down_sample import DownSample
from modeling.deepSfp.up_sample import UpSample
from modeling.deepSfp.final_layer import FinalLayer

class deepSfp(nn.Module):
    def __init__(self,input_channel,device):
        super(deepSfp, self).__init__()
        self.feature_extractor = FeatureExtractor(input_channel=input_channel,output_channel=4)
        self.down_sample_1 = DownSample(input_channel=4,output_channel=32)
        self.down_sample_2 = DownSample(input_channel=32,output_channel=64)
        self.down_sample_3 = DownSample(input_channel=64,output_channel=128)
        self.down_sample_4 = DownSample(input_channel=128,output_channel=256)
        self.down_sample_5 = DownSample(input_channel=256,output_channel=512)

        self.up_sample_5 = UpSample(input_channel=512,output_channel=512,isFirst=True)
        self.up_sample_4 = UpSample(input_channel=512,output_channel=256)
        self.up_sample_3 = UpSample(input_channel=256,output_channel=128)
        self.up_sample_2 = UpSample(input_channel=128,output_channel=64)
        self.up_sample_1 = UpSample(input_channel=64,output_channel=32)

        self.final_layer = FinalLayer(input_channel=32)


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

    def forward(self, polar,prior):
        img  = polar
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
        # network

        x = torch.concat((polar,prior,atten_map),dim=1)

        # method
        # x= torch.concat((polar,prior),dim=1)




        x0 = self.feature_extractor(x)

        # down sample
        x1 = self.down_sample_1(x0)
        x2 = self.down_sample_2(x1)
        x3 = self.down_sample_3(x2)
        x4 = self.down_sample_4(x3)
        x5 = self.down_sample_5(x4)

        # up sample
        polar_16 = F.interpolate(polar,scale_factor= 1/16.0)
        x = self.up_sample_5(x5,x5,polar_16)
        polar_8 = F.interpolate(polar,scale_factor= 1/8.0)
        x = self.up_sample_4(x,x4,polar_8)
        polar_4 = F.interpolate(polar,scale_factor=1/4.0)
        x = self.up_sample_3(x,x3,polar_4)
        polar_2 = F.interpolate(polar,scale_factor=1/2.0)
        x = self.up_sample_2(x,x2,polar_2)
        x = self.up_sample_1(x,x1,polar)

        x = self.final_layer(x)


        return x,x


if __name__ == '__main__':
    prior = torch.ones([8,3,256,256])
    polar = torch.ones([8,4,256,256])

    model = deepSfp(input_channel=3)
    res = model(polar,prior)
    print(res.shape)