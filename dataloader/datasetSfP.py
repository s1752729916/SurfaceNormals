# -*-coding:utf-8-*-
import os
import glob
import sys
import time
import numpy
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from imgaug import augmenters as iaa
import imgaug as ia
import imageio
import pandas as pd
import API.utils
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# from API.utils import exr_loader
class SfPSurfaceDataset(Dataset):
    """
    Dataset class for testing model on estimation of surface normals.
    Uses imgaug for image augmentations.
    If a label_dir is blank ( None, ''), it will assume labels do not exist and return a tensor of zeros
    for the label.
    Args:
        input_I_sum_dir (str): Path to folder containing the I_sum imgs(.png format)
        input_normal_dir (str): Root dir  to folder containing the four input normal dirs (synthesis-normal-0,-1,-2,-3) (.exr format)
        label_dir (str): (Optional) Path to folder containing the labels (.exr format).
                         If no labels exists, pass empty string ('') or None.
        mask_dir (str): (Optional) Path to folder containing the masks (.png format).
                         If no labels exists, pass empty string ('') or None.
        transform (imgaug transforms): imgaug Transforms to be applied to the imgs
        input_only (list, str): List of transforms that are to be applied only to the input img
    """

    def __init__(self,root_dir,list_file, transform=None, input_only=None):
        self.root_dir = root_dir
        self.transform = transform
        self.input_only = input_only
        self.list_file = list_file
        # Create list of filenames
        self._datalist_polar_0 = []
        self._datalist_polar_45 = []
        self._datalist_polar_90 = []
        self._datalist_polar_135 = []
        self._datalist_dolp = []
        self._datalist_aolp = []
        self._datalist_input_normal_0 = []
        self._datalist_input_normal_1 = []
        self._datalist_input_normal_2 = []
        self._datalist_input_normal_3 = []
        self._datalist_label = []
        self._datalist_mask = []
        self._create_lists_filenames()

    def __len__(self):
        return len(self._datalist_label)

    def __getitem__(self, index):
        '''Returns an item from the dataset at the given index. If no labels directory has been specified,
        then a tensor of zeroes will be returned as the label.
        Args:
            index (int): index of the item required from dataset.
        Returns:
            torch.Tensor: Tensor of input image, shape is (13, Height, Width), the 13 channels are (norm_0 x 3 , norm_1 x 3, norm_2 x 3, norm_3 x 3, intensity)
            torch.Tensor: Tensor of label (Tensor of zeroes is labels_dir is "" or None), the shape (3, Height, Width) for real norm
            torch.Tensor: Tensor of mask ( The shape (3, 1, Height, Width) for real norm
        '''

        # -- 1、prepare paths
        # all_time = time.time()
        polar_0_path = self._datalist_polar_0[index]
        polar_45_path = self._datalist_polar_45[index]
        polar_90_path = self._datalist_polar_90[index]
        polar_135_path = self._datalist_polar_135[index]

        norm_0_path = self._datalist_input_normal_0[index]
        norm_1_path = self._datalist_input_normal_1[index]
        norm_2_path = self._datalist_input_normal_2[index]
        norm_3_path = self._datalist_input_normal_3[index]
        if not os.path.exists(norm_0_path):
            raise ValueError('This file not exists: {} '.format(self.norm_0_path))
        dolpPath = self._datalist_dolp[index]
        aolpPath = self._datalist_aolp[index]

        mask_path = self._datalist_mask[index]
        label_path = self._datalist_label[index]
        if not os.path.exists(label_path):
            raise ValueError('This file not exists: {} '.format(self.label_path))
        # start = time.time()
        # print(index)
        # print(I_sum_path)
        # print(norm_0_path)
        # print(norm_1_path)
        # print(norm_2_path)
        # print(norm_3_path)
        # print(mask_path)
        # print(label_path)
        # print('')
        # -- 2、load imgs
        # I_0 = imageio.imread(polar_0_path)  # numpy array shape is (height, width)
        # I_45 = imageio.imread(polar_45_path)  # numpy array shape is (height, width)
        # I_90 = imageio.imread(polar_90_path)  # numpy array shape is (height, width)
        # I_135 = imageio.imread(polar_135_path)  # numpy array shape is (height, width)

        dolpImg = API.utils.rgb_loader(dolpPath)  # numpy array shape is (height, width, 3)
        aolpImg = API.utils.rgb_loader(aolpPath)

        norm_0 = API.utils.rgb_loader(norm_0_path)  # numpy array shape is (height, width, 3)
        norm_1 = API.utils.rgb_loader(norm_1_path)
        norm_2 = API.utils.rgb_loader(norm_2_path)
        norm_3 = API.utils.rgb_loader(norm_3_path)
        mask_img = API.utils.mask_loader(mask_path)
        label_img = API.utils.rgb_loader(label_path).transpose(2, 0, 1) # To( C,H,W)
        # print("load time",time.time()-start)

        # -- 3、concat inputs
        # start = time.time()
        height = label_img.shape[1]
        width = label_img.shape[2]
        input_img_arr = numpy.zeros([14, height, width], dtype=np.uint8)  # shape is (13 x H x W)
        input_img_arr[0,:,:] = dolpImg
        input_img_arr[1,:,:] = aolpImg
        input_img_arr[2:5, :, :] = norm_0.transpose(2, 0, 1)  # 3 x H x W
        input_img_arr[5:8, :, :] = norm_1.transpose(2, 0, 1)
        input_img_arr[8:11, :, :] = norm_2.transpose(2, 0, 1)
        input_img_arr[11:14, :, :] = norm_3.transpose(2, 0, 1)


        # print("concat time",time.time()-start)

        # -- 5、apply mask to inputs and label
        input_img_arr[:, mask_img == 0] = 0
        label_img[:, mask_img == 0] = 0

        # print("apply mask time",time.time()-start)

        # -- 4、apply image augmentations
        # start = time.time()

        if self.transform:
            # apply augment to inputs
            det_tf = self.transform.to_deterministic()
            input_img_arr = det_tf.augment_image(
                input_img_arr.transpose(1, 2, 0))  # augment_image require shape (H, W, C)
            input_img_arr = input_img_arr.transpose(2, 0, 1)  # To Shape: (13, H, W)

            # apply augment to label
            label_img = det_tf.augment_image(label_img.transpose(1, 2, 0), hooks=ia.HooksImages(
                activator=self._activator_masks))  # some transforms only appy to inputs, not label
            label_img = label_img.transpose(2, 0, 1)  # To (C,H,W)
            # apply mask
            mask_img = det_tf.augment_image(mask_img, hooks=ia.HooksImages(
                activator=self._activator_masks))  # some transforms only appy to inputs, not label
        # print("augmentation time",time.time()-start)

        # -- 4、normalize
        # start = time.time()

        input_tensor = transforms.ToTensor()(
            input_img_arr.copy().transpose(1, 2, 0))  # ToTensor contains the normalization process

        # convert 0-255 to (-1,1)
        label_tensor = torch.from_numpy(label_img).float()  # (C,H,W)
        label_tensor = (label_tensor-127)/127.0
        label_tensor = nn.functional.normalize(label_tensor,p=2,dim=0)
        mask_tensor = torch.from_numpy(mask_img.copy()).unsqueeze(0)

        # print("normalize time",time.time()-start)
        # print("total time:" ,time.time()-all_time)
        # print("input shape:",input_tensor.shape)
        # print("label_tensor:",label_tensor.shape)
        # print("mask_tensor:",mask_tensor.shape)

        # fig = plt.figure()
        # ax0 = plt.subplot(611)
        # ax0.imshow(label_img.transpose(1,2,0))
        # ax1 = plt.subplot(612)
        # ax1.imshow(input_img_arr[1:4,:,:].transpose(1,2,0))
        # ax2 = plt.subplot(613)
        # ax2.imshow(input_img_arr[4:7,:,:].transpose(1,2,0))
        # ax3 = plt.subplot(614)
        # ax3.imshow(input_img_arr[7:10,:,:].transpose(1,2,0))
        # ax4 = plt.subplot(615)
        # ax4.imshow(input_img_arr[10:13,:,:].transpose(1,2,0))
        # ax5 = plt.subplot(616)
        # ax5.imshow(input_img_arr[0,:,:])
        # plt.show()
        # print("getitem time consumeption:",time.time()-start)
        # print("label_tensor shape:",label_tensor.shape)
        a = torch.split(input_tensor,[2,12],dim=0)
        polar_tensor = a[0]
        normal_tensor = a[1]

        return polar_tensor,normal_tensor, label_tensor, mask_tensor


    def _create_lists_filenames(self):
        '''Creates a list of filenames of images and labels each in dataset
        The label at index N will match the image at index N.
        Args: None
        Raises:
            ValueError: If the given directories are invalid
            ValueError: No images were found in given directory
            ValueError: Number of images and labels do not match
        '''

        #-- 1.读取list文件
        list_data = pd.read_csv(self.list_file)
        data_array = np.array(list_data).reshape(-1)
        # 然后转化为list形式
        list_data = data_array.tolist()
        num = len(list_data)
        for i in range(0,num):
            sub_root,suffix = list_data[i].split('/')
            I_sum_dir = os.path.join(os.path.join(os.path.join(self.root_dir,sub_root),'I-Sum'),suffix+'.png')
            I_0_dir = os.path.join(os.path.join(os.path.join(self.root_dir,sub_root),'I-0'),suffix+'.png')
            I_45_dir = os.path.join(os.path.join(os.path.join(self.root_dir,sub_root),'I-45'),suffix+'.png')
            I_90_dir = os.path.join(os.path.join(os.path.join(self.root_dir,sub_root),'I-90'),suffix+'.png')
            I_135_dir = os.path.join(os.path.join(os.path.join(self.root_dir,sub_root),'I-135'),suffix+'.png')
            normal_dir = os.path.join(os.path.join(os.path.join(self.root_dir,sub_root),'normals-png'),suffix+'.png')
            mask_dir = os.path.join(os.path.join(os.path.join(self.root_dir,sub_root),'masks'),suffix+'.png')
            dolp_dir = os.path.join(os.path.join(os.path.join(os.path.join(self.root_dir,sub_root),'params'),'DoLP'),suffix+'.png')
            aolp_dir = os.path.join(os.path.join(os.path.join(os.path.join(self.root_dir,sub_root),'params'),'AoLP'),suffix+'.png')

            synthesis_normal_0_dir = os.path.join(os.path.join(os.path.join(os.path.join(self.root_dir,sub_root),'synthesis-normals'),'synthesis-normal-0'),suffix+'.png')
            synthesis_normal_1_dir = os.path.join(os.path.join(os.path.join(os.path.join(self.root_dir,sub_root),'synthesis-normals'),'synthesis-normal-1'),suffix+'.png')
            synthesis_normal_2_dir = os.path.join(os.path.join(os.path.join(os.path.join(self.root_dir,sub_root),'synthesis-normals'),'synthesis-normal-2'),suffix+'.png')
            synthesis_normal_3_dir = os.path.join(os.path.join(os.path.join(os.path.join(self.root_dir,sub_root),'synthesis-normals'),'synthesis-normal-3'),suffix+'.png')

            self._datalist_polar_0.append(I_0_dir)
            self._datalist_polar_45.append(I_45_dir)
            self._datalist_polar_90.append(I_90_dir)
            self._datalist_polar_135.append(I_135_dir)
            self._datalist_label.append(normal_dir)
            self._datalist_input_normal_0.append(synthesis_normal_0_dir)
            self._datalist_input_normal_1.append(synthesis_normal_1_dir)
            self._datalist_input_normal_2.append(synthesis_normal_2_dir)
            self._datalist_input_normal_3.append(synthesis_normal_3_dir)
            self._datalist_mask.append(mask_dir)
            self._datalist_dolp.append(dolp_dir)
            self._datalist_aolp.append(aolp_dir)

        # -- input polar images
        numISumImages = len(self._datalist_polar_0)
        if numISumImages == 0:
            raise ValueError('No rgb images found in given directory. Searched in dir: {} '.format(self.input_polar_dir))
        numISumImages = len(self._datalist_polar_45)
        if numISumImages == 0:
            raise ValueError('No rgb images found in given directory. Searched in dir: {} '.format(self.input_polar_dir))
        numISumImages = len(self._datalist_polar_90)
        if numISumImages == 0:
            raise ValueError('No rgb images found in given directory. Searched in dir: {} '.format(self.input_polar_dir))
        numISumImages = len(self._datalist_polar_135)
        if numISumImages == 0:
            raise ValueError('No rgb images found in given directory. Searched in dir: {} '.format(self.input_polar_dir))

        numDoLPImages = len(self._datalist_dolp)
        if numDoLPImages == 0:
            raise ValueError('No dolp images found in given directory')
        numAoLPImages = len(self._datalist_aolp)
        if numAoLPImages == 0:
            raise ValueError('No aolp images found in given directory')

        # -- input normal images
        numNorm_0 = len(self._datalist_input_normal_0)
        if numNorm_0 == 0:
            raise ValueError('No input normal_0 files found in given directory. ')

        numNorm_1 = len(self._datalist_input_normal_1)
        if numNorm_1 == 0:
            raise ValueError('No input normal_1 files found in given directory. ')
        numNorm_2 = len(self._datalist_input_normal_2)
        if numNorm_2 == 0:
            raise ValueError('No input normal_2 files found in given directory. ')

        numNorm_3 = len(self._datalist_input_normal_3)
        if numNorm_3 == 0:
            raise ValueError('No input normal_3 files found in given directory. ')
        if not (numNorm_0 == numNorm_1 == numNorm_2 == numNorm_3):
            raise ValueError('Numbers of input normals are different.')


        # -- 3、labels(real normals)
        numLabels = len(self._datalist_label)
        if numLabels == 0:
            raise ValueError(
                'No input label files found in given directory' )

        # -- 4、masks
        numMasks = len(self._datalist_mask)
        if numMasks == 0:
            raise ValueError(
                'No input mask files found in given directory. ')

        # -- 5、verify number of every input
        if not (numISumImages == numNorm_0 == numLabels == numLabels == numDoLPImages == numAoLPImages):
            print(numISumImages, numNorm_0, numNorm_1, numNorm_2, numNorm_3, numMasks, numLabels,numDoLPImages,numAoLPImages)
            raise ValueError('Numbers of inputs(rgb,normal,label,mask,DoLP,AoLP) are different.')


    def _activator_masks(self, images, augmenter, parents, default):
        '''Used with imgaug to help only apply some augmentations to images and not labels
        Eg: Blur is applied to input only, not label. However, resize is applied to both.
        '''
        if self.input_only and augmenter.name in self.input_only:
            return False
        else:
            return default


if (__name__ == '__main__'):
    import matplotlib.pyplot as plt

    from torch.utils.data import DataLoader
    from torchvision import transforms
    import torchvision

    # Example Augmentations using imgaug
    imsize = 512
    augs_train = iaa.Sequential([
        # Geometric Augs
        iaa.Resize((imsize, imsize)),  # Resize image
        # iaa.Fliplr(0.5),
        # iaa.Flipud(0.5),
        # iaa.Rot90((0, 4)),

        # Blur and Noise
        # iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 1.5), name="gaus-blur")),
        # iaa.Sometimes(0.1, iaa.Grayscale(alpha=(0.0, 1.0), from_colorspace="RGB", name="grayscale")),
        # iaa.Sometimes(0.2, iaa.AdditiveLaplaceNoise(scale=(0, 0.1*255), per_channel=True, name="gaus-noise")),

        # Color, Contrast, etc.
        # iaa.Sometimes(0.2, iaa.Multiply((0.75, 1.25), per_channel=0.1, name="brightness")),
        # iaa.Sometimes(0.2, iaa.GammaContrast((0.7, 1.3), per_channel=0.1, name="contrast")),
        # iaa.Sometimes(0.2, iaa.AddToHueAndSaturation((-20, 20), name="hue-sat")),
        # iaa.Sometimes(0.3, iaa.Add((-20, 20), per_channel=0.5, name="color-jitter")),
    ])
    augs_test = iaa.Sequential([
        # Geometric Augs
        iaa.Resize((imsize, imsize), 0),
    ])

    augs = augs_train
    input_only = ["gaus-blur", "grayscale", "gaus-noise", "brightness", "contrast", "hue-sat", "color-jitter"]
    root_dir = '/media/zjw/smq/data/DeepSfP/pngs'
    list_file = '/media/zjw/smq/data/DeepSfP/pngs/test_list.csv'
    dt_train = SfPSurfaceDataset(root_dir,list_file, transform = augs_train)

    polar_tensor,normal_tensor, label_tensor, mask_tensor = dt_train.__getitem__(11)


