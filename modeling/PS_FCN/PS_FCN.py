import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from . import model_utils

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
        self.deconv1 = model_utils.conv(batchNorm, 128, 128,  k=3, stride=1, pad=1)
        self.deconv2 = model_utils.conv(batchNorm, 128, 128,  k=3, stride=1, pad=1)
        self.deconv3 = model_utils.deconv(128, 64)
        self.est_normal= self._make_output(64, 3, k=3, stride=1, pad=1)
        self.other   = other

    def _make_output(self, cin, cout, k=3, stride=1, pad=1):
        return nn.Sequential(
               nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False))

    def forward(self, x, shape):
        x      = x.view(shape[0], shape[1], shape[2], shape[3])
        out    = self.deconv1(x)
        out    = self.deconv2(out)
        out    = self.deconv3(out)
        normal = self.est_normal(out)
        normal = torch.nn.functional.normalize(normal, 2, 1)
        return normal

class PS_FCN(nn.Module):
    def __init__(self, fuse_type='max', batchNorm=False, c_in=3, other={},device = None):
        super(PS_FCN, self).__init__()
        self.extractor = FeatExtractor(batchNorm, c_in, other)
        self.regressor = Regressor(batchNorm, other)
        self.c_in      = c_in
        self.fuse_type = fuse_type
        self.other = other

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.kernel_size = 9
        self.lamda = 1
        self.m = 0.5
        self.mean_kernel = torch.ones([1,1,self.kernel_size,self.kernel_size])/self.kernel_size**2
        self.mean_kernel = self.mean_kernel.to(device)
        self.mean_kernel = nn.Parameter(data=self.mean_kernel, requires_grad=False)
        self.sum_kernel_1 = torch.ones([1,1,self.kernel_size,self.kernel_size])
        self.sum_kernel_1 = self.sum_kernel_1.to(device)
        self.sum_kernel_1 = nn.Parameter(data=self.sum_kernel_1, requires_grad=False)

        self.sum_kernel_3 = torch.ones([3,3,self.kernel_size,self.kernel_size])
        self.sum_kernel_3 = self.sum_kernel_3.to(device)
        self.sum_kernel_3 = nn.Parameter(data=self.sum_kernel_3, requires_grad=False)

    def forward(self, params,normals):

        img  = params
        img_split = torch.split(img, 1, 1)
        aolp = img_split[1]

        # get attention map
        mean_map = nn.functional.conv2d(aolp,self.mean_kernel,padding=self.kernel_size//2)
        abs_map =torch.abs(aolp - mean_map)
        abs_map = torch.pow(abs_map,self.m)
        atten_map = nn.functional.conv2d(abs_map,self.sum_kernel_1,padding=self.kernel_size//2)
        shape = atten_map.shape
        atten_map = torch.reshape(atten_map,[shape[0],-1])
        max_values,indices = torch.max(atten_map,dim = 1)
        max_values = torch.reshape(max_values,[shape[0],1])
        atten_map = torch.div(atten_map,max_values)
        atten_map = torch.reshape(atten_map,[shape[0],shape[1],shape[2],shape[3]])

        param0 = torch.cat((params,atten_map),dim = 1)

        img   = normals
        img_split = torch.split(img, 3, 1)


        feats = []
        feat, shape = self.extractor(param0)
        feats.append(feat)
        for i in range(len(img_split)):
            net_in = img_split[i]
            feat, shape = self.extractor(net_in)
            feats.append(feat)
        if self.fuse_type == 'mean':
            feat_fused = torch.stack(feats, 1).mean(1)
        elif self.fuse_type == 'max':
            feat_fused, _ = torch.stack(feats, 1).max(1)
        normal = self.regressor(feat_fused, shape)
        return normal,normal