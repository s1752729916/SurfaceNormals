import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
import modeling.PS_FCN.model_utils as model_utils
import modeling.utils.SKBlock as SKBlock
import modeling.utils.ESPABlock as ESPABlock
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.decoder_masks import build_decoder
from modeling.backbone import build_backbone
class FeatExtractor(nn.Module):
    def __init__(self, batchNorm=False, c_in=3, other={}):
        super(FeatExtractor, self).__init__()
        self.other = other
        self.conv1 = model_utils.conv(batchNorm, c_in, 64,  k=3, stride=1, pad=1)
        self.conv2 = model_utils.conv(batchNorm, 64,   128, k=3, stride=2, pad=1)
        self.conv3 = model_utils.conv(batchNorm, 128,  128, k=3, stride=1, pad=1)
        self.conv4 = model_utils.conv(batchNorm, 128,  256, k=3, stride=2, pad=1)
        self.conv5 = model_utils.conv(batchNorm, 256,  256, k=3, stride=1, pad=1)
        self.conv6 = model_utils.deconv(256, 128)
        self.conv7 = model_utils.conv(batchNorm, 128, 128, k=3, stride=1, pad=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out_feat = self.conv7(out)
        n, c, h, w = out_feat.data.shape
        out_feat   = out_feat.view(-1)
        return out_feat, [n, c, h, w]

class Regressor(nn.Module):
    def __init__(self, batchNorm=False, other={}):
        super(Regressor, self).__init__()
        self.other   = other
        self.deconv1 = model_utils.conv(batchNorm, 256, 128,  k=3, stride=1, pad=1)
        self.deconv2 = model_utils.conv(batchNorm, 128, 128,  k=3, stride=1, pad=1)
        self.deconv3 = model_utils.deconv(128, 64)
        self.est_normal= self._make_output(64, 3, k=3, stride=1, pad=1)
        self.other   = other

    def _make_output(self, cin, cout, k=3, stride=1, pad=1):
        return nn.Sequential(
               nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False))

    def forward(self, x, shape):
        out    = self.deconv1(x)
        out    = self.deconv2(out)
        out    = self.deconv3(out)
        normal = self.est_normal(out)
        normal = torch.nn.functional.normalize(normal, 2, 1)
        return normal

class smqNet(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(smqNet, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone_orig = build_backbone(in_channels=2,backbone=backbone, output_stride=output_stride, BatchNorm=BatchNorm)
        self.aspp_orig = build_aspp(backbone, output_stride, BatchNorm)
        self.backbone_prior = build_backbone(in_channels=12,backbone=backbone, output_stride=output_stride, BatchNorm=BatchNorm)
        self.aspp_prior = build_aspp(backbone,output_stride,BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm,double=True)
        if freeze_bn:
            self.freeze_bn()

    def forward(self, orig,prior):
        x_orig, low_level_feat_orig = self.backbone_orig(orig)
        x_orig = self.aspp_orig(x_orig)

        x_prior, low_level_feat_prior = self.backbone_prior(prior)
        x_prior = self.aspp_prior(x_prior)

        x = torch.cat((x_orig,x_prior),1)
        low_level_feat = torch.cat((low_level_feat_orig,low_level_feat_prior),1)

        x = self.decoder(x, low_level_feat)

        x = F.interpolate(x, size=orig.size()[2:], mode='bilinear', align_corners=True)
        # x = torch.sin(x)
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()