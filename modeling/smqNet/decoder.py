import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.smqNet.upsample import UpSample
from modeling.deepSfp.deepSfp import FinalLayer
class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm,double = False):
        super(Decoder, self).__init__()
        if backbone == 'resnet50' or backbone== 'resnet101' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        last_channel = 256
        if(double):
            low_level_inplanes = low_level_inplanes*2
            last_channel = last_channel*2

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        last_channels = last_channel + 48

        # self.upsample_128 = UpSample(input_channel=last_channels,output_channel=128)
        self.upsample_256 = UpSample(input_channel=last_channels,output_channel=128)
        self.upsample_512 = UpSample(input_channel=128,output_channel=64)
        self.final_layer = FinalLayer(input_channel=64)


        self._init_weight()


    def forward(self, x, low_level_feat,input):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        # input_4 = F.interpolate(input,scale_factor= 1/4.0)  # 128x128
        input_2 = F.interpolate(input,scale_factor= 1/2.0)  # 256x256
        # x = self.upsample_128(x,input_4)
        x = self.upsample_256(x,input_2)
        x = self.upsample_512(x,input)
        x = self.final_layer(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(num_classes, backbone, BatchNorm,double = False):
    return Decoder(num_classes, backbone, BatchNorm,double)