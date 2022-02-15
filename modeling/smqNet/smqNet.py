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
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

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
                 sync_bn=True, freeze_bn=False,device= None):
        super(smqNet, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.kernel_size = 9
        self.lamda = 1
        self.mean_kernel = torch.ones([1,1,self.kernel_size,self.kernel_size])/self.kernel_size**2
        self.mean_kernel = self.mean_kernel.to(device)
        self.sum_kernel_1 = torch.ones([1,1,self.kernel_size,self.kernel_size])
        self.sum_kernel_1 = self.sum_kernel_1.to(device)
        self.sum_kernel_3 = torch.ones([3,3,self.kernel_size,self.kernel_size])
        self.sum_kernel_3 = self.sum_kernel_3.to(device)

        self.backbone_orig = build_backbone(in_channels=2,backbone=backbone, output_stride=output_stride, BatchNorm=BatchNorm)
        self.aspp_orig = build_aspp(backbone, output_stride, BatchNorm)
        self.backbone_prior = build_backbone(in_channels=12,backbone=backbone, output_stride=output_stride, BatchNorm=BatchNorm)
        self.aspp_prior = build_aspp(backbone,output_stride,BatchNorm)
        self.decoder_orig = build_decoder(num_classes, backbone, BatchNorm,double=False)
        self.decoder_prior =  build_decoder(num_classes, backbone, BatchNorm,double=False)
        self.backbone_atten = build_backbone(in_channels=1,backbone=backbone,output_stride=output_stride,BatchNorm=BatchNorm)
        self.aspp_atten = build_aspp(backbone,output_stride,BatchNorm)
        self.decoder_atten = build_decoder(num_classes, backbone, BatchNorm,double=False)
        self.decoder_test = build_decoder(1,backbone,BatchNorm,double=False)


        self.sigmoid = nn.Sigmoid()
        if freeze_bn:
            self.freeze_bn()
        self.attention_layer_x = ESPABlock.EPSABlock(inplanes=512,planes=512)
        self.attention_layer_low_level_feat = ESPABlock.EPSABlock(inplanes=512,planes=512)

        self.writer = SummaryWriter()
    def forward(self, orig,prior):
        img   = orig
        img_split = torch.split(img, 1, 1)
        aolp = img_split[1]

        # get attention map

        mean_map = nn.functional.conv2d(aolp,self.mean_kernel,padding=self.kernel_size//2)
        abs_map = torch.abs(aolp - mean_map)
        atten_map = nn.functional.conv2d(abs_map,self.sum_kernel_1,padding=self.kernel_size//2)
        shape = atten_map.shape
        atten_map = torch.reshape(atten_map,[shape[0],-1])
        max_values,indices = torch.max(atten_map,dim = 1)
        max_values = torch.reshape(max_values,[shape[0],1])
        atten_map = torch.div(atten_map,max_values)
        atten_map = torch.reshape(atten_map,[shape[0],shape[1],shape[2],shape[3]])

        # atten_map_cpu = atten_map.detach().cpu()
        # plt.imshow(atten_map_cpu[0,:,:,:].squeeze(0))
        # plt.show()





        x_orig, low_level_feat_orig = self.backbone_orig(orig)
        x_orig = self.aspp_orig(x_orig)
        # normal_orig = self.decoder_orig(x_orig,low_level_feat_orig)
        # normal_orig = F.interpolate(normal_orig, size=orig.size()[2:], mode='bilinear', align_corners=True)
        # normal_orig = nn.functional.normalize(normal_orig, p=2, dim=1)

        x_prior, low_level_feat_prior = self.backbone_prior(prior)
        x_prior = self.aspp_prior(x_prior)

        # normal_prior = self.decoder_prior(x_prior,low_level_feat_prior)
        # normal_prior = F.interpolate(normal_prior, size=orig.size()[2:], mode='bilinear', align_corners=True)
        # normal_prior = nn.functional.normalize(normal_prior, p=2, dim=1)

        # cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        # attention_map  = 1.0 - cos(normal_orig, normal_prior)
        # attention_map = torch.unsqueeze(attention_map,dim=1)
        # normal_atten = attention_map*normal_prior + (1-attention_map)*normal_orig

        # x_atten = torch.cat((normal_orig,normal_prior,attention_map),dim = 1)

        x_atten,low_level_feat_atten = self.backbone_atten(atten_map)
        x_atten = self.aspp_atten(x_atten)
        x_atten = self.sigmoid(x_atten)
        low_level_feat_atten = self.sigmoid(low_level_feat_atten)

        x_fusion = x_atten*x_orig + (1-x_atten)*x_prior
        low_level_feat_fusion = low_level_feat_atten*low_level_feat_orig + (1-low_level_feat_atten)*low_level_feat_prior

        # x_atten,low_level_feat_atten = self.backbone_atten(x_atten)
        # x_atten = self.aspp_atten(x_atten)
        x = self.decoder_atten(x_fusion,low_level_feat_fusion)
        # atten_map = self.decoder_test(x_atten,low_level_feat_atten)
        # atten_map = torch.sin(atten_map)
        # atten_map = self.sigmoid(atten_map)

        # x = torch.cat((x_orig,x_prior),1)
        # low_level_feat = torch.cat((low_level_feat_orig,low_level_feat_prior),1)
        # x = (1-x_atten)*x_orig + x_atten*x_prior
        # low_level_feat = (1-low_level_feat_atten)*low_level_feat_orig + low_level_feat_atten*low_level_feat_prior

        # channel attention operation
        # x = self.attention_layer_x(x)
        # low_level_feat = self.attention_layer_low_level_feat(low_level_feat)
        x = F.interpolate(x, size=orig.size()[2:], mode='bilinear', align_corners=True)

        # atten_map = F.interpolate(atten_map, size=orig.size()[2:], mode='bilinear', align_corners=True)


        # # smooth process
        # p = 1-atten_map
        # temp = p*x
        # N_neightbor = nn.functional.conv2d(temp,self.sum_kernel_3,padding=self.kernel_size//2)
        # p_sum = nn.functional.conv2d(p,self.sum_kernel_1,padding=self.kernel_size//2)
        # N_neightbor = N_neightbor/p_sum
        #
        # N = p*x + self.lamda*(1-p)*N_neightbor

        # x = torch.cat([x,atten_map],dim=1)
        # x,low_level_feat = self.backbone_test(x)
        # x = self.aspp_test(x)
        # x = self.decoder_test(x,low_level_feat)
        # x = F.interpolate(x, size=orig.size()[2:], mode='bilinear', align_corners=True)

        return x,atten_map

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()