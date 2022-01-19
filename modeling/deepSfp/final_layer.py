import torch
import torch.nn as nn
import torch.nn.functional as F

class FinalLayer(nn.Module):
    def __init__(self,input_channel):
        super(FinalLayer, self).__init__()
        ks = 3
        pw = ks//2
        self.conv1 = nn.Conv2d(in_channels=input_channel,out_channels=input_channel,kernel_size=ks,padding=pw)
        self.instace_norm = nn.InstanceNorm2d(num_features=input_channel)
        self.leaky_relu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(in_channels=input_channel,out_channels=3,kernel_size=1)





    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.instace_norm(x1)
        x1 = self.leaky_relu(x1)
        x = x1 + x
        x = self.conv2(x)

        return x



if __name__ == '__main__':
    x = torch.ones([8,32,256,256])


    model = FinalLayer(input_channel=32)
    res = model(x)
    print(res.shape)