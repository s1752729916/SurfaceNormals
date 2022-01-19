import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    def __init__(self,input_channel = 7, output_channel = 4):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channel,out_channels=input_channel,kernel_size=1)
        self.leaky_relu = nn.LeakyReLU()
        self.instance_norm1 = nn.InstanceNorm2d(num_features = input_channel)
        self.conv2 = nn.Conv2d(in_channels=input_channel,out_channels = output_channel,kernel_size=1)
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
    x = torch.ones([8,7,256,256])


    model = FeatureExtractor()
    res = model(x)
    print(res.shape)