# -*-coding:utf-8-*-
import os
import glob
import sys
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


    def __getitem__(self, index):
        '''Returns an item from the dataset at the given index. If no labels directory has been specified,
        then a tensor of zeroes will be returned as the label.
        Args:
            index (int): index of the item required from dataset.
        Returns:
            torch.Tensor: Tensor of input image
            torch.Tensor: Tensor of label (Tensor of zeroes is labels_dir is "" or None)
        '''
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
        input_normal_0_search_str = os.path.join(input_normal_0_search_str,'*-synthsis-normal.exr')
        self._datalist_input_normal_0 = sorted(glob.glob(input_normal_0_search_str))
        numNorm_0 = len(self._datalist_input_normal_0)
        if numNorm_0 == 0:
            raise ValueError('No input normal_0 files found in given directory. Searched in dir: {} '.format(input_normal_0_search_str))

        input_normal_1_search_str = os.path.join(self.input_norm_dir,'synthesis-normal-1')
        input_normal_1_search_str = os.path.join(input_normal_1_search_str,'*-synthsis-normal.exr')
        self._datalist_input_normal_1 = sorted(glob.glob(input_normal_1_search_str))
        numNorm_1 = len(self._datalist_input_normal_1)
        if numNorm_1 == 0:
            raise ValueError('No input normal_1 files found in given directory. Searched in dir: {} '.format(input_normal_1_search_str))

        input_normal_2_search_str = os.path.join(self.input_norm_dir,'synthesis-normal-2')
        input_normal_2_search_str = os.path.join(input_normal_2_search_str,'*-synthsis-normal.exr')
        self._datalist_input_normal_2 = sorted(glob.glob(input_normal_2_search_str))
        numNorm_2 = len(self._datalist_input_normal_2)
        if numNorm_2 == 0:
            raise ValueError('No input normal_2 files found in given directory. Searched in dir: {} '.format(input_normal_2_search_str))

        input_normal_3_search_str = os.path.join(self.input_norm_dir,'synthesis-normal-3')
        input_normal_3_search_str = os.path.join(input_normal_3_search_str,'*-synthsis-normal.exr')
        self._datalist_input_normal_3 = sorted(glob.glob(input_normal_3_search_str))
        numNorm_3 = len(self._datalist_input_normal_3)
        if numNorm_3 == 0:
            raise ValueError('No input normal_3 files found in given directory. Searched in dir: {} '.format(input_normal_3_search_str))
        if not (numNorm_0==numNorm_1==numNorm_2==numNorm_3):
            raise ValueError('Numbers of input normals are different.')

        #-- 3、labels(real normals)
        labels_search_str = os.path.join(self.labels_dir,'*-cameraNormals.exr')
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
            raise ValueError('Numbers of inputs(rgb,normal,label,mask) are different.')










if(__name__ == '__main__'):
    dt_train = SurfaceNormalsDataset(input_rgb_dir='/media/smq/移动硬盘/学习/数据集/ClearGrasp/cleargrasp-dataset-train/cup-with-waves-train/rgb-imgs',
                                     input_normal_dir='/media/smq/移动硬盘/学习/数据集/ClearGrasp/cleargrasp-dataset-train/cup-with-waves-train/synthesis-normals',
                                     label_dir='/media/smq/移动硬盘/学习/数据集/ClearGrasp/cleargrasp-dataset-train/cup-with-waves-train/camera-normals',
                                     mask_dir='/media/smq/移动硬盘/学习/数据集/ClearGrasp/cleargrasp-dataset-train/cup-with-waves-train/segmentation-masks')
