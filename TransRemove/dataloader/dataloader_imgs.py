# -*-coding:utf-8-*-
import os
import glob
import sys
import time
import numpy
import psutil.tests.test_bsd
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from imgaug import augmenters as iaa
import imgaug as ia
import imageio

import API.utils
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# from API.utils import exr_loader
class SurfaceDataset(Dataset):
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

    def __init__(self,  root_dir,label_dir,dolp_label,aolp_label,mask_dir=None, transform=None, input_only=None):
        self.I_0_dir = os.path.join(root_dir,'I-0')
        self.I_45_dir = os.path.join(root_dir,'I-45')
        self.I_90_dir = os.path.join(root_dir,'I-90')
        self.I_135_dir = os.path.join(root_dir,'I-135')
        self.dolp_label = dolp_label
        self.aolp_label = aolp_label
        self.labels_dir = label_dir
        self.masks_dir = mask_dir
        self.transform = transform
        self.input_only = input_only

        # Create list of filenames
        self._datalist_I_0 = []
        self._datalist_I_45 = []
        self._datalist_I_90 = []
        self._datalist_I_135 = []
        self._datalist_aolp_label = []
        self._datalist_dolp_label = []
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
        I_0_path = self._datalist_I_0[index]
        I_45_path = self._datalist_I_45[index]
        I_90_path = self._datalist_I_90[index]
        I_135_path = self._datalist_I_135[index]
        dolplabelPath = self._datalist_dolp_label[index]
        aolplabelPath = self._datalist_aolp_label[index]


        if(self.masks_dir is not None):
            mask_path = self._datalist_mask[index]
        label_path = self._datalist_label[index]
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
        I_0_img = API.utils.rgb_loader(I_0_path)  # numpy array shape is (height, width, 3)
        I_45_img = API.utils.rgb_loader(I_45_path)
        I_90_img = API.utils.rgb_loader(I_90_path)
        I_135_img = API.utils.rgb_loader(I_135_path)

        dolplabelImg = API.utils.rgb_loader(dolplabelPath)
        aolplabelImg = API.utils.rgb_loader(aolplabelPath)
        mask_img = API.utils.mask_loader(mask_path)
        label_img = API.utils.rgb_loader(label_path).transpose(2, 0, 1) # To( C,H,W)
        # print("load time",time.time()-start)

        # -- 3、concat inputs
        # start = time.time()
        height = label_img.shape[1]
        width = label_img.shape[2]
        input_img_arr = numpy.zeros([4, height, width], dtype=np.uint8)  # shape is (3 x H x W)
        input_img_arr[0, :, :] = I_0_img
        input_img_arr[1, :, :] = I_45_img
        input_img_arr[2, :, :] = I_90_img
        input_img_arr[3, :, :] = I_135_img

        output_params_arr = numpy.zeros([2, height, width], dtype=np.uint8)  # shape is (3 x H x W)
        output_params_arr[0,:,:] = dolplabelImg
        output_params_arr[1,:,:] = aolplabelImg


        # print("concat time",time.time()-start)

        # -- 5、apply mask to inputs and label
        if(self.masks_dir is not None):
            # start = time.time()
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

            output_params_arr = det_tf.augment_image(output_params_arr.transpose(1, 2, 0), hooks=ia.HooksImages(
                activator=self._activator_masks))  # some transforms only appy to inputs, not label
            output_params_arr = output_params_arr.transpose(2, 0, 1)  # To (C,H,W)

            # apply mask
            if(self.masks_dir is not None):
                mask_img = det_tf.augment_image(mask_img, hooks=ia.HooksImages(
                    activator=self._activator_masks))  # some transforms only appy to inputs, not label
        # print("augmentation time",time.time()-start)

        # -- 4、normalize
        # start = time.time()

        input_tensor = transforms.ToTensor()(
            input_img_arr.copy().transpose(1, 2, 0))  # ToTensor contains the normalization process
        # convert dolp aolp to (0,1)
        output_params_tensor = transforms.ToTensor()(output_params_arr.copy().transpose(1,2,0))

        # convert 0-255 to (-1,1)
        label_tensor = torch.from_numpy(label_img).float()  # (C,H,W)
        label_tensor = (label_tensor-127)/127.0
        label_tensor = nn.functional.normalize(label_tensor,p=2,dim=0)

        if(self.masks_dir is not None):
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
        if(self.masks_dir is not None):
            return input_tensor,output_params_tensor, label_tensor, mask_tensor
        else:
            return input_tensor,output_params_tensor,label_tensor


    def _create_lists_filenames(self):
        '''Creates a list of filenames of images and labels each in dataset
        The label at index N will match the image at index N.
        Args: None
        Raises:
            ValueError: If the given directories are invalid
            ValueError: No images were found in given directory
            ValueError: Number of images and labels do not match
        '''
        assert os.path.isdir(self.labels_dir), 'Dataloader given labels directory that does not exist: "%s"' % (self.labels_dir)
        if(self.masks_dir is not None):
            assert os.path.isdir(
                self.masks_dir), 'Dataloader given masks_dir images directory that does not exist: "%s"' % (self.masks_dir)


        # -- input I_sum images

        # -- input DoLP & AoLP images
        I_0_search_str = os.path.join(self.I_0_dir, '*.png')
        self._datalist_I_0 = sorted(glob.glob(I_0_search_str))
        numI0 = len(self._datalist_I_0)
        if numI0 == 0:
            raise ValueError('No input I-0 files found in given directory. Searched in dir: {} '.format(
                self._datalist_I_0))

        I_45_search_str = os.path.join(self.I_45_dir, '*.png')
        self._datalist_I_45 = sorted(glob.glob(I_45_search_str))
        numI45 = len(self._datalist_I_45)
        if numI45 == 0:
            raise ValueError('No input I-45 files found in given directory. Searched in dir: {} '.format(
                I_45_search_str))

        I_90_search_str = os.path.join(self.I_90_dir, '*.png')
        self._datalist_I_90 = sorted(glob.glob(I_90_search_str))
        numI90 = len(self._datalist_I_90)
        if numI90 == 0:
            raise ValueError('No input I-90 files found in given directory. Searched in dir: {} '.format(
                I_45_search_str))

        I_135_search_str = os.path.join(self.I_135_dir, '*.png')
        self._datalist_I_135 = sorted(glob.glob(I_135_search_str))
        numI135 = len(self._datalist_I_135)
        if numI135 == 0:
            raise ValueError('No input I-90 files found in given directory. Searched in dir: {} '.format(
                I_135_search_str))

        # -- 3、labels(real normals and label params)
        labels_search_str = os.path.join(self.labels_dir, '*.png')
        self._datalist_label = sorted(glob.glob(labels_search_str))
        numLabels = len(self._datalist_label)
        if numLabels == 0:
            raise ValueError(
                'No input label files found in given directory. Searched in dir: {} '.format(self.labels_dir))

        dolp_label_search_str = os.path.join(self.dolp_label,'*.png')
        self._datalist_dolp_label = sorted(glob.glob(dolp_label_search_str))
        numDolplabels = len(self._datalist_dolp_label)
        if numDolplabels == 0:
            raise ValueError(-
                'No input dolp label files found in given directory. Searched in dir: {} '.format(self.dolp_label))

        aolp_label_search_str = os.path.join(self.aolp_label,'*.png')
        self._datalist_aolp_label = sorted(glob.glob(aolp_label_search_str))
        numAolplabels = len(self._datalist_aolp_label)
        if numAolplabels == 0:
            raise ValueError(
                'No input aolp label files found in given directory. Searched in dir: {} '.format(self.aolp_label))
        # -- 4、masks
        if(self.masks_dir is not None):
            masks_search_str = os.path.join(self.masks_dir, '*.png')
            self._datalist_mask = sorted(glob.glob(masks_search_str))
            numMasks = len(self._datalist_mask)
            if numMasks == 0:
                raise ValueError(
                    'No input mask files found in given directory. Searched in dir: {} '.format(self.masks_dir))

        # -- 5、verify number of every input
            if not (numI0==numI45==numI90==numI135 == numMasks == numLabels == numAolplabels ==numDolplabels):
                print(numI0, numI45,numI90,numI135, numMasks, numLabels)
                raise ValueError('Numbers of inputs(dolp,aolp,label,mask, dolplabel,aolplabel) are different.')


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

    dt_train = SurfaceDataset(
        root_dir='/media/disk2/smq_data/samples/End2End2/hemi-sphere-big-1-20',
        dolp_label = '/media/disk2/smq_data/samples/End2End2/hemi-sphere-big-1-20/standard-params/DoLP',
        aolp_label = '/media/disk2/smq_data/samples/End2End2/hemi-sphere-big-1-20/standard-params/AoLP',
        label_dir = '/media/disk2/smq_data/samples/End2End2/hemi-sphere-big-1-20/normals-png',
        mask_dir= '/media/disk2/smq_data/samples/End2End2/hemi-sphere-big-1-20/masks')

    # print("dataset")
    # batch_size = 16
    # testloader = DataLoader(dt_train, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True,prefetch_factor=2)
    # print("dataloader")
    # # Show 1 Shuffled Batch of Images
    # import loss_functions
    # for ii, batch in enumerate(testloader):
    #     # Get Batch
    #     input_tensor, label_tensor,mask_tensor = batch
    #     print("ii:",ii)
    #
    #     print(" ")
    #     print(" ")
    #     print("input_vec:", input_tensor[4:7, :, :].shape)
    #     print("target_vec:", label_tensor.shape)
    #     print("mask_vec:", mask_tensor.squeeze(1).shape)
    #     print('loss', loss_functions.loss_fn_cosine(input_vec=input_tensor[:,10:13, :, :],
    #                                                 target_vec=label_tensor,
    #                                                 mask_tensor=mask_tensor.squeeze(1),
    #                                                 reduction='elementwise_mean'))
    #     loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3 = loss_functions.metric_calculator_batch(
    #         input_tensor[:,4:7, :, :], label_tensor.double(), mask_tensor.squeeze(1))
    #     print("loss_deg_mean:",loss_deg_mean)
    #     print("loss_deg_median:",loss_deg_median)
    #     print("percentage_1:",percentage_1)
    #     print("percentage_2:",percentage_2)
    #     print("percentage_3:",percentage_3)
    #
    #     # # Show Batch
    #     # sample = torch.cat((img[1:,:,:], label), 5)
    #     # im_vis = torchvision.utils.make_grid(sample, nrow=batch_size // 4, padding=2, normalize=True, scale_each=True)
    #     # plt.imshow(im_vis.numpy().transpose(1, 2, 0))
    #     # plt.show()
    #
    #     # break

    import loss_functions

    input_tensor,output_params_tensor, label_tensor, mask_tensor = dt_train.__getitem__(11)
    input_img_arr = input_tensor.numpy()
    output_params_img = output_params_tensor.numpy()
    label_img = label_tensor.numpy()
    mask_img = mask_tensor.numpy()
    fig = plt.figure()
    ax0 = plt.subplot(241)
    ax0.imshow(label_img.transpose(1, 2, 0))
    ax1 = plt.subplot(245)
    ax1.imshow(input_img_arr[0, :, :])
    ax2 = plt.subplot(246)
    ax2.imshow(input_img_arr[1, :, :])
    ax3 = plt.subplot(247)
    ax3.imshow(output_params_tensor[0, :, :])
    ax4 = plt.subplot(248)
    ax4.imshow(output_params_tensor[1, :, :])
    # print("mask_valid:_nums:", len(np.where(mask_img > 0)[1]))
    # print(mask_img.shape)
    # print("input_vec:", input_tensor[4:7, :, :].unsqueeze(0).shape)
    # print("target_vec:", label_tensor.unsqueeze(0).shape)
    # print("mask_vec:", mask_tensor.unsqueeze(0).shape)
    # print('loss', loss_functions.loss_fn_cosine(input_vec=input_tensor[1:4, :, :].unsqueeze(0),
    #                                             target_vec=label_tensor.unsqueeze(0),
    #                                             mask_tensor=mask_tensor.unsqueeze(0).squeeze(1),
    #                                             reduction='elementwise_mean'))
    # loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3 = loss_functions.metric_calculator_batch(
    #     input_tensor[10:13, :, :].unsqueeze(0), label_tensor.double().unsqueeze(0))
    # print("loss_deg_mean:", loss_deg_mean)
    # print("loss_deg_median:", loss_deg_median)
    # print("percentage_1:", percentage_1)
    # print("percentage_2:", percentage_2)
    # print("percentage_3:", percentage_3)

    plt.show()