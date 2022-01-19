import torch
import torch.nn as nn
import torch.nn.functional as F
from SPADE import SPADE
class UpSample(nn.Module):
    def __init__(self,input_channel, output_channel,isFirst = False):
        super(UpSample, self).__init__()
        ks = 3
        pw = ks//2
        self.SPADE1 = SPADE(input_channel = input_channel*2)
        self.SPADE2 = SPADE(input_channel = output_channel)
        self.leaky_relu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(in_channels=input_channel*2,out_channels=output_channel,kernel_size=ks,padding=pw)
        self.conv2 = nn.Conv2d(in_channels=output_channel,out_channels=output_channel,kernel_size=ks,padding=pw)
        self.isFirst = isFirst
        self.conv_c = nn.Conv2d(in_channels=input_channel//2,out_channels=input_channel,kernel_size=1) # 将下采样的特征通道数进行适配





    def forward(self, x,x2,input):
        if(not self.isFirst):
            x2 = self.conv_c(x2)
        x = F.interpolate(x,size=input.size()[2:],mode='bilinear')
        x2 = F.interpolate(x2,input.size()[2:],mode='bilinear')
        x = torch.cat((x,x2),dim=1)
        x = self.SPADE1(x,input)
        x = self.leaky_relu(x)
        x_temp = self.conv1(x)
        x = self.SPADE2(x_temp,input)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = x + x_temp

        return x



if __name__ == '__main__':
    x = torch.ones([8,512,8,8])
    x1 = torch.ones([8,512,8,8])
    input = torch.ones(8,4,16,16)
    model = UpSample(512,512)
    res = model(x,x1,input)
    print(res.shape)