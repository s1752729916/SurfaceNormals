import torch
import torch.nn as nn
import torch.nn.functional as F

class SPADE(nn.Module):
    def __init__(self,input_channel):
        super(SPADE, self).__init__()
        npolar = 4
        nhidden = 128
        ks = 3
        pw = ks//2
        self.batch_norm = nn.BatchNorm2d(input_channel)
        self.conv = nn.Conv2d(in_channels=input_channel,out_channels=input_channel,kernel_size=ks)



        self.polar_conv1 = nn.Conv2d(in_channels=npolar,out_channels=nhidden,kernel_size=ks,padding=pw)
        self.polar_conv_alpha = nn.Conv2d(in_channels=nhidden,out_channels=input_channel,kernel_size=ks,padding=pw)
        self.polar_conv_beta = nn.Conv2d(in_channels=nhidden,out_channels=input_channel,kernel_size=ks,padding=pw)

        self.relu = nn.ReLU()




    def forward(self, x,input):
        x = self.batch_norm(x)
        x1 = self.polar_conv1(input)
        x1 = self.relu(x1)
        alpha = self.polar_conv_alpha(x1)
        beta = self.polar_conv_beta(x1)
        x = x + x*alpha + beta

        return x



if __name__ == '__main__':
    x = torch.ones([8,512,16,16])
    input = torch.ones([8,4,16,16])

    model = SPADE(512)
    res = model(x,input)
    print(res.shape)