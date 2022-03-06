""" Full assembly of the parts to form the complete network """
import torch

from modeling.UNet.unet_parts import *
from modeling.attention.SelfAttention import ScaledDotProductAttention

class SPWBlock(nn.Module):
    def __init__(self,c_in,heads,H,W):
        super(SPWBlock, self).__init__()

        self.c = c_in
        self.atten = ScaledDotProductAttention(d_k=c_in,d_v=c_in,d_model=c_in,h=heads)
        self.layer_norm = nn.LayerNorm([c_in,H,W])
    def forward(self, x):
        # x (B,C,H,W)
        x = self.layer_norm(x)
        bs, c, h, w = x.shape
        y = x.view(bs, c, -1).permute(0, 2, 1)  # bs,h*w,c
        y,attn = self.atten(y,y,y)  # bs,h*w,c
        y = y.permute(0,2,1).view(bs,c,h,w)

        return y




class SPW(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False,device = None):
        super(SPW, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)


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


        # self attention in BottleNeck
        self.attention_layers = self._make_self_attention(c_in=512,H=32,W =32,heads=8,n = 8)

        # view encoding
        # method compare
        rows = torch.arange(0,512,1)
        cols = torch.arange(0,512,1)
        z = torch.ones([512,512])
        rows = (rows/511)*2-1
        cols = (cols/511)*2-1
        x,y = torch.meshgrid(rows,cols)
        V = torch.ones([6,3,512,512])

        V[:,0,:,:] = x
        V[:,1,:,:] = y
        V[:,2,:,:] = z
        self.V1 = torch.ones([1,3,512,512])
        self.V1[0,:,:,:] = V[0,:,:,:]
        self.V = V.to(device)
        self.V1 = self.V1.to(device)
        self.V = nn.Parameter(data=self.V,requires_grad=False)
        self.V1 = nn.Parameter(data=self.V1,requires_grad=False)


    def _make_self_attention(self,c_in,H,W,heads,n):
        layers = []
        for i in range(0,n):
            layers.append(SPWBlock(c_in=c_in,H=H,W=W,heads=heads))
        return nn.Sequential(*layers)




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


        # method compare
        # bs,c,h,w = params.shape
        # if(bs >1):
        #     x = torch.cat((params,self.V),dim=1)
        # else:
        #     x = torch.cat((params, self.V1), dim=1)
        # network compare
        x = torch.cat((params,normals,atten_map),dim=1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4) # (B,512,32,32)
        x5 = self.attention_layers(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits,logits

if __name__ =='__main__':
    x = torch.ones([6,512,32,32])
    model = SPWBlock(c_in=512,H=32,W=32,heads=1)
    y = model(x)
    print(y)