import torch
import torch.nn as nn
import torch.nn.functional as F

class DownSample(nn.Module):
    def __init__(self,input_channel, output_channel):
        super(DownSample, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channel,out_channels=input_channel,kernel_size=3,stride=2,padding=3)
        self.leaky_relu = nn.LeakyReLU()
        self.instance_norm1 = nn.InstanceNorm2d(num_features = input_channel)
        self.conv2 = nn.Conv2d(in_channels=input_channel,out_channels = output_channel,kernel_size=3)
        self.instance_norm2 = nn.InstanceNorm2d(num_features = output_channel)



    def forward(self, x):
        x = self.conv1(x)
        x = self.instance_norm1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.instance_norm2(x)
        x = self.leaky_relu(x)
        return x



if __name__ == '__main__':
    x = torch.ones([8,4,256,256])


    model = DownSample(4,32)
    res = model(x)
    print(res.shape)