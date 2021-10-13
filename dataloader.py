# -*-coding:utf-8-*-
import os
import glob
import sys
import time
import numpy
from PIL import Image
import Imath
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from imgaug import augmenters as iaa
import imgaug as ia
import imageio

import API.utils
from API.utils import exr_loader
class SurfaceNormalsDataset(Dataset):
    """
    Dataset class for training model on estimation of surface normals.
    Uses imgaug for image augmentations.
    If a label_dir is blank ( None, ''), it will assume labels do not exist and return a tensor of zeros
    for the label.
    Args:
        input_rgb_dir (str): Path to folder containing the inpur rgb images (.jpg format)
        input_normal_dir (str): Root dir  to folder containing the four input normal dirs (synthesis-normal-0,-1,-2,-3) (.exr format)
        label_dir (str): (Optional) Path to folder containing the labels (.exr format).
                         If no labels exists, pass empty string ('') or None.
        mask_dir (str): (Optional) Path to folder containing the masks (.png format).
                         If no labels exists, pass empty string ('') or None.
        transform (imgaug transforms): imgaug Transforms to be applied to the imgs
        input_only (list, str): List of transforms that are to be applied only to the input img
    """
    def __init__(self,input_rgb_dir,input_normal_dir,label_dir,mask_dir,transform=None,input_only=None):
        self.input_rgb_dir = input_rgb_dir  # rgb path is used to calculate intensity image(rgb2grey)
        self.input_norm_dir = input_normal_dir
        self.labels_dir = label_dir
        self.masks_dir = mask_dir
        self.transform = transform
        self.input_only = input_only

        # Create list of filenames
        self._datalist_input_normal_0= []
        self._datalist_input_normal_1= []
        self._datalist_input_normal_2= []
        self._datalist_input_normal_3= []
        self._datalist_input_rgb = []
        self._datalist_mask = []
        self._datalist_label = []
        self._create_lists_filenames()
    def __len__(self):
        return len(self._datalist_input_rgb)


    def __getitem__(self, index):
        '''Returns an item from the dataset at the given index. If no labels directory has been specified,
        then a tensor of zeroes will be returned as the label.
        Args:
            index (int): index of the item required from dataset.
        Returns:
            torch.Tensor: Tensor of input image, shape is (13, Height, Width), the 13 channels are (norm_0 x 3 , norm_1 x 3, norm_2 x 3, norm_3 x 3, intensity)
            torch.Tensor: Tensor of label (Tensor of zeroes is labels_dir is "" or None), the shape (3, Height, Width) for real norm
            torch.Tensor: Tensor of mask
        '''
        #-- 1、prepare paths
        # all_time = time.time()
        rgb_img_path = self._datalist_input_rgb[index]
        norm_0_path = self._datalist_input_normal_0[index]
        norm_1_path = self._datalist_input_normal_1[index]
        norm_2_path = self._datalist_input_normal_2[index]
        norm_3_path = self._datalist_input_normal_3[index]
        mask_path = self._datalist_mask[index]
        label_path = self._datalist_label[index]
        # start = time.time()
        #-- 2、load imgs
        rgb_img = imageio.imread(rgb_img_path)  # numpy array shape is (height, width, 3)
        grey_img = API.utils.rgb2grey(rgb_img)  # numpy array shape is (height, width)
        norm_0 = API.utils.rgb_loader(norm_0_path)  #numpy array shape is (height, width, 3)
        norm_1 = API.utils.rgb_loader(norm_1_path)
        norm_2 = API.utils.rgb_loader(norm_2_path)
        norm_3 = API.utils.rgb_loader(norm_3_path)
        mask_img = API.utils.mask_loader(mask_path)
        label_img = API.utils.rgb_loader(label_path).transpose(2,0,1)
        # print("load time",time.time()-start)



       #-- 3、concat inputs
        # start = time.time()
        height = grey_img.shape[0]
        width = grey_img.shape[1]
        input_img_arr = numpy.zeros([13,height,width],dtype = np.uint8)    #  shape is (13 x H x W)
        input_img_arr[0,:,:] = grey_img
        input_img_arr[1:4,:,:] = norm_0.transpose(2,0,1)  #3 x H x W
        input_img_arr[4:7,:,:] = norm_1.transpose(2,0,1)
        input_img_arr[7:10,:,:] = norm_2.transpose(2,0,1)
        input_img_arr[10:13,:,:] = norm_3.transpose(2,0,1)
        # print("concat time",time.time()-start)

        #-- 5、apply mask to inputs and label
        # start = time.time()
        input_img_arr[:,mask_img==0] = 0
        label_img[:,mask_img==0] = 0

        # print("apply mask time",time.time()-start)

        #-- 4、apply image augmentations
        # start = time.time()

        if self.transform:
            # apply augment to inputs
            det_tf = self.transform.to_deterministic()
            input_img_arr = det_tf.augment_image(input_img_arr.transpose(1,2,0))  #augment_image require shape (H, W, C)
            input_img_arr = input_img_arr.transpose(2,0,1)  # To Shape: (13, H, W)

            # apply augment to label
            label_img = det_tf.augment_image(label_img.transpose(1,2,0), hooks=ia.HooksImages(activator=self._activator_masks))  # some transforms only appy to inputs, not label
            label_img = label_img.transpose(2,0,1)
            # apply mask
            mask_img = det_tf.augment_image(mask_img, hooks=ia.HooksImages(activator=self._activator_masks))  # some transforms only appy to inputs, not label
        # print("augmentation time",time.time()-start)

        #-- 4、normalize
        # start = time.time()

        input_tensor = transforms.ToTensor()(input_img_arr.copy().transpose(1,2,0))  #ToTensor contains the normalization process
        label_tensor = transforms.ToTensor()(label_img.copy().transpose(1,2,0))
        mask_tensor = torch.from_numpy(mask_img)

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
        return input_tensor,label_tensor,mask_tensor


    def _create_lists_filenames(self):
        '''Creates a list of filenames of images and labels each in dataset
        The label at index N will match the image at index N.
        Args: None
        Raises:
            ValueError: If the given directories are invalid
            ValueError: No images were found in given directory
            ValueError: Number of images and labels do not match
        '''
        assert os.path.isdir(self.input_rgb_dir), 'Dataloader given rgb images directory that does not exist: "%s"' % (self.input_rgb_dir)
        assert os.path.isdir(self.input_norm_dir), 'Dataloader given input norm  directory that does not exist: "%s"' % (self.input_norm_dir)
        assert os.path.isdir(self.labels_dir), 'Dataloader given labels directory that does not exist: "%s"' % (self.labels_dir)
        assert os.path.isdir(self.masks_dir), 'Dataloader given masks_dir images directory that does not exist: "%s"' % (self.masks_dir)

        #-- 1、input rgb images
        rgb_path_search_str = os.path.join(self.input_rgb_dir,'*'+'-rgb.jpg')
        self._datalist_input_rgb = sorted(glob.glob(rgb_path_search_str))
        numRgbImages = len(self._datalist_input_rgb)
        if numRgbImages == 0:
            raise ValueError('No rgb images found in given directory. Searched in dir: {} '.format(self.input_rgb_dir))
        
        #-- 2、input normal images
        input_normal_0_search_str = os.path.join(self.input_norm_dir,'synthesis-normal-0')
        input_normal_0_search_str = os.path.join(input_normal_0_search_str,'*-synthsis-normal.png')
        self._datalist_input_normal_0 = sorted(glob.glob(input_normal_0_search_str))
        numNorm_0 = len(self._datalist_input_normal_0)
        if numNorm_0 == 0:
            raise ValueError('No input normal_0 files found in given directory. Searched in dir: {} '.format(input_normal_0_search_str))

        input_normal_1_search_str = os.path.join(self.input_norm_dir,'synthesis-normal-1')
        input_normal_1_search_str = os.path.join(input_normal_1_search_str,'*-synthsis-normal.png')
        self._datalist_input_normal_1 = sorted(glob.glob(input_normal_1_search_str))
        numNorm_1 = len(self._datalist_input_normal_1)
        if numNorm_1 == 0:
            raise ValueError('No input normal_1 files found in given directory. Searched in dir: {} '.format(input_normal_1_search_str))

        input_normal_2_search_str = os.path.join(self.input_norm_dir,'synthesis-normal-2')
        input_normal_2_search_str = os.path.join(input_normal_2_search_str,'*-synthsis-normal.png')
        self._datalist_input_normal_2 = sorted(glob.glob(input_normal_2_search_str))
        numNorm_2 = len(self._datalist_input_normal_2)
        if numNorm_2 == 0:
            raise ValueError('No input normal_2 files found in given directory. Searched in dir: {} '.format(input_normal_2_search_str))

        input_normal_3_search_str = os.path.join(self.input_norm_dir,'synthesis-normal-3')
        input_normal_3_search_str = os.path.join(input_normal_3_search_str,'*-synthsis-normal.png')
        self._datalist_input_normal_3 = sorted(glob.glob(input_normal_3_search_str))
        numNorm_3 = len(self._datalist_input_normal_3)
        if numNorm_3 == 0:
            raise ValueError('No input normal_3 files found in given directory. Searched in dir: {} '.format(input_normal_3_search_str))
        if not (numNorm_0==numNorm_1==numNorm_2==numNorm_3):
            raise ValueError('Numbers of input normals are different.')

        #-- 3、labels(real normals)
        labels_search_str = os.path.join(self.labels_dir,'*-cameraNormals.png')
        self._datalist_label = sorted(glob.glob(labels_search_str))
        numLabels = len(self._datalist_label)
        if numLabels==0:
            raise ValueError('No input label files found in given directory. Searched in dir: {} '.format(self.labels_dir))

        #-- 4、masks
        masks_search_str = os.path.join(self.masks_dir,'*-segmentation-mask.png')
        self._datalist_mask = sorted(glob.glob(masks_search_str))
        numMasks = len(self._datalist_mask)
        if numMasks==0:
            raise ValueError('No input mask files found in given directory. Searched in dir: {} '.format(self.masks_dir))


        #-- 5、verify number of every input
        if not (numRgbImages==numNorm_0==numLabels==numLabels):
            print(numRgbImages,numNorm_0,numNorm_1,numNorm_2,numNorm_3,numMasks,numLabels)
            raise ValueError('Numbers of inputs(rgb,normal,label,mask) are different.')

    def _activator_masks(self, images, augmenter, parents, default):
        '''Used with imgaug to help only apply some augmentations to images and not labels
        Eg: Blur is applied to input only, not label. However, resize is applied to both.
        '''
        if self.input_only and augmenter.name in self.input_only:
            return False
        else:
            return default







if(__name__ == '__main__'):
    import matplotlib.pyplot as plt

    from torch.utils.data import DataLoader
    from torchvision import transforms
    import torchvision

    # Example Augmentations using imgaug
    imsize = 512
    augs_train = iaa.Sequential([
        # Geometric Augs
        iaa.Resize((imsize, imsize)), # Resize image
        # iaa.Fliplr(0.5),
        # iaa.Flipud(0.5),
        # iaa.Rot90((0, 4)),

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
    augs_test = iaa.Sequential([
        # Geometric Augs
        iaa.Resize((imsize, imsize), 0),
    ])

    augs = augs_train
    input_only =  ["gaus-blur", "grayscale", "gaus-noise", "brightness", "contrast", "hue-sat", "color-jitter"]
    dt_train = SurfaceNormalsDataset(input_rgb_dir='/media/smq/移动硬盘/学习/数据集/ClearGrasp/cleargrasp-dataset-train/flower-bath-bomb-train/rgb-imgs',
                                     input_normal_dir='/media/smq/移动硬盘/学习/数据集/ClearGrasp/cleargrasp-dataset-train/flower-bath-bomb-train/synthesis-normals',
                                     label_dir='/media/smq/移动硬盘/学习/数据集/ClearGrasp/cleargrasp-dataset-train/flower-bath-bomb-train/camera-normals',
                                     mask_dir='/media/smq/移动硬盘/学习/数据集/ClearGrasp/cleargrasp-dataset-train/flower-bath-bomb-train/segmentation-masks',transform=augs_train,input_only=input_only)
    print("dataset")
    batch_size = 16
    testloader = DataLoader(dt_train, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True,prefetch_factor=2)
    print("dataloader")
    # Show 1 Shuffled Batch of Images
    # for ii, batch in enumerate(testloader):
    #     # Get Batch
    #     img, label,mask = batch
    #     print("ii:",ii)
    #     print('image shape, type: ', img.shape, img.dtype)
    #     print('label shape, type: ', label.shape, label.dtype)
    #     print('mask shape, type: ', mask.shape, mask.dtype)
    #     print(" ")
    #     print(" ")
    #
    #     # # Show Batch
    #     # sample = torch.cat((img[1:,:,:], label), 5)
    #     # im_vis = torchvision.utils.make_grid(sample, nrow=batch_size // 4, padding=2, normalize=True, scale_each=True)
    #     # plt.imshow(im_vis.numpy().transpose(1, 2, 0))
    #     # plt.show()
    #
    #     # break



    input_tensor, label_tensor,mask_tensor = dt_train.__getitem__(10)
    input_img_arr = input_tensor.numpy()
    label_img = label_tensor.numpy()
    mask_img = mask_tensor.numpy()
    fig = plt.figure()
    ax0 = plt.subplot(241)
    ax0.imshow(label_img.transpose(1,2,0))
    ax1 = plt.subplot(245)
    ax1.imshow(input_img_arr[1:4,:,:].transpose(1,2,0))
    ax2 = plt.subplot(246)
    ax2.imshow(input_img_arr[4:7,:,:].transpose(1,2,0))
    ax3 = plt.subplot(247)
    ax3.imshow(input_img_arr[7:10,:,:].transpose(1,2,0))
    ax4 = plt.subplot(248)
    ax4.imshow(input_img_arr[10:13,:,:].transpose(1,2,0))
    ax5 = plt.subplot(242)
    ax5.imshow(input_img_arr[0,:,:])
    ax6 = plt.subplot(243)
    print(mask_img.shape)
    ax6.imshow(mask_img)
    plt.show()