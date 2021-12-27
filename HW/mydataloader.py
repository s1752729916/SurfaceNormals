import os
import glob
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from imgaug import augmenters as iaa
import imgaug as ia
import imageio
class AnimalFruitsDataset(Dataset):
    def __init__(self, img_dir, label_number,transform = None,input_only = None):
        """
        动物、水果数据集合的Dataset类
        Args:
            img_dir (str): 图片路径
            label_number (int): 该类图片所对应的类别号，自己定义
            transform (imgaug transforms): 数据增强
            input_only (list, str): 只对输入使用的数据增强
        """
        self.img_dir = img_dir
        self.label_number = label_number
        self.transform = transform
        self.input_only = input_only
        self._imgs_list_ = []
        self._create_lists_filenames()

    def _create_lists_filenames(self):
        assert os.path.isdir(self.img_dir), 'Dataloader given img_dir directory that does not exist: "%s"' % (
            self.img_dir)
        search_str = os.path.join(self.img_dir,"*.png")
        self._imgs_list_ = sorted(glob.glob(search_str))

    def __getitem__(self, item):
        img = imageio.imread(self._imgs_list_[item],pilmode="RGB") #默认是(H,W,C)格式
        # 添加
        if self.transform:
            # apply augment to inputs
            det_tf = self.transform.to_deterministic()

            img = det_tf.augment_image(img)  # augment_image require shape (H, W, C)

        input_tensor = transforms.ToTensor()(img.copy())  # ToTensor contains the normalization process
        return input_tensor,self.label_number
    def __len__(self):
        return len(self._imgs_list_)

if (__name__ == '__main__'):
    imsize = 283
    augs_train = iaa.Sequential([
        # Geometric Augs
        # iaa.Resize((imsize, imsize)), # Resize image
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Rot90((0, 4)),

        # Blur and Noise
        #iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 1.5), name="gaus-blur")),
        #iaa.Sometimes(0.1, iaa.Grayscale(alpha=(0.0, 1.0), from_colorspace="RGB", name="grayscale")),
        # iaa.Sometimes(0.2, iaa.AdditiveLaplaceNoise(scale=(0, 0.1*255), per_channel=True, name="gaus-noise")),

        # Color, Contrast, etc.
        #iaa.Sometimes(0.2, iaa.Multiply((0.75, 1.25), per_channel=0.1, name="brightness")),
        # iaa.Sometimes(0.2, iaa.GammaContrast((0.7, 1.3), per_channel=0.1, name="contrast")),
        # iaa.Sometimes(0.2, iaa.AddToHueAndSaturation((-20, 20), name="hue-sat")),
        #iaa.Sometimes(0.3, iaa.Add((-20, 20), per_channel=0.5, name="color-jitter")),
    ])


    augs = augs_train
    input_only =  ["gaus-blur", "grayscale", "gaus-noise", "brightness", "contrast", "hue-sat", "color-jitter"]
    dt_train = AnimalFruitsDataset('/home/zjw/smq/SurfaceNormals/HW/problem1/Data/动物/动物/狗',label_number=0,transform=augs_train,input_only=input_only)
    dt_train.__getitem__(10)

