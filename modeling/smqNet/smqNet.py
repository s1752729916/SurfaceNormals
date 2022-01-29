import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
import modeling.PS_FCN.model_utils as model_utils
import modeling.utils.SKBlock as SKBlock
import modeling.utils.ESPABlock as ESPABlock
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
    def __init__(self, fuse_type='max', batchNorm=True, c_orig = 2,c_prior = 3, other={}):
        super(smqNet, self).__init__()
        self.extractor_orig = FeatExtractor(batchNorm,c_orig,other)
        self.extractor_prior = FeatExtractor(batchNorm, c_prior, other)
        self.decoder = Regressor(batchNorm, other)
        self.fuse_type = fuse_type
        self.attention_layer = ESPABlock.EPSABlock(inplanes=256,planes=256)


    def forward(self, params,synthesis_normals):

        normals_split = torch.split(synthesis_normals, 3, 1)

        # extractor_prior
        feat_orig,shape = self.extractor_orig(params)

        feat_orig = feat_orig.view(shape[0], shape[1], shape[2], shape[3])

        # extractor_orig
        feats = []
        for i in range(len(normals_split)):
            net_in = normals_split[i]
            feat, shape = self.extractor_prior(net_in)
            feats.append(feat)
        if self.fuse_type == 'mean':
            feat_prior = torch.stack(feats, 1).mean(1)
        elif self.fuse_type == 'max':
            feat_prior, _ = torch.stack(feats, 1).max(1)
        feat_prior =  feat_prior.view(shape[0], shape[1], shape[2], shape[3])

        # concat features of orig and prior
        # features = self.attention_layer(feat_orig,feat_prior)
        features = torch.cat((feat_orig,feat_prior),dim=1)
        # features = self.attention_layer(features)
        normal = self.decoder(features, shape)

        return normal