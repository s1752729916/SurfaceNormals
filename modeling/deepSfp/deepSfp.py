'''
implemetaion of <deep shape from polarization,ECCV,2020>
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from feature_extractor import FeatureExtractor
from down_sample import DownSample
from up_sample import UpSample
from final_layer import FinalLayer

class deepSfp(nn.Module):
    def __init__(self):
        super(deepSfp, self).__init__()
        self.feature_extractor = FeatureExtractor(input_channel=7,output_channel=4)
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


    def forward(self, polar,prior):
        x = torch.concat((prior,polar),dim=1)
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


        return x


if __name__ == '__main__':
    prior = torch.ones([8,3,256,256])
    polar = torch.ones([8,4,256,256])

    model = deepSfp()
    res = model(polar,prior)
    print(res.shape)