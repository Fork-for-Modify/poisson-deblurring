import sys
import torch.distributed as dist
from torch.utils import data
from torch.utils.data import Dataset, DataLoader, DistributedSampler, random_split
from scipy import ndimage
import cv2
import os
import numpy as np
from tqdm import tqdm
from os.path import join as opj


# =================
# loading single frame and blur it
# =================

# =================
# basic functions
# =================


class BlurImgDataset_Exp_all2CPU(Dataset):
    """
    load blurry image, kernel, and ground truth (for 'simu' exp) for normal experiments, load entire dataset to CPU to speed the data load process
    exp_mode:
        - simu: with gt
        - real: no gt   
    patch_sz: assign image size of patch processing to save GPU memory, default = None, use whole image (TODO: patch processing and stitching)
    """

    def __init__(self, blur_img_dir, psf_dir, gt_dir=None, patch_sz=None, exp_mode='simu'):
        super(BlurImgDataset_Exp_all2CPU, self).__init__()
        self.patch_sz = [patch_sz] * \
            2 if isinstance(patch_sz, int) else patch_sz
        # use loaded psf, rather than generated
        self.img_dir, self.psf_dir, self.gt_dir = blur_img_dir, psf_dir, gt_dir
        self.exp_mode = exp_mode
        self.imgs = []
        self.psfs = []
        self.gts = []

        # get image paths and load images
        img_paths = []
        img_names = sorted(os.listdir(blur_img_dir))
        img_paths = [opj(blur_img_dir, img_name) for img_name in img_names]
        self.img_num = len(img_paths)

        for img_path in tqdm(img_paths, desc='Loading image to CPU'):

            if img_path.split('.')[-1].lower() not in ['jpg', 'jpeg', 'png', 'tif', 'bmp']:
                print('Skip a non-image file: %s' % (img_path))
                continue
            img = cv2.imread(img_path)
            assert img is not None, 'Image read falied'
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.imgs.append(img)

        # get gt paths and load gts
        if self.exp_mode == 'simu':
            gt_paths = []
            gt_names = sorted(os.listdir(gt_dir))
            gt_paths = [opj(gt_dir, gt_name) for gt_name in gt_names]
            self.gt_num = len(gt_paths)

            for gt_path in tqdm(gt_paths, desc='Loading gt to CPU'):
                if gt_path.split('.')[-1].lower() not in ['jpg', 'jpeg', 'png', 'tif', 'bmp']:
                    print('Skip a non-image file: %s' % (gt_path))
                    continue
                gt = cv2.imread(gt_path)
                assert gt is not None, 'Image read falied'
                gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
                self.gts.append(gt)

        # get loaded psf paths and load psfs
        psf_paths = []
        psf_names = sorted(os.listdir(psf_dir))
        psf_paths = [opj(psf_dir, psf_name) for psf_name in psf_names]
        self.psf_num = len(psf_names)

        for psf_path in tqdm(psf_paths, desc='Loading psf to CPU'):
            if psf_path.split('.')[-1].lower() not in ['jpg', 'jpeg', 'png', 'tif', 'bmp']:
                print('Skip a non-image file: %s' % psf_path)
                continue
            psf = cv2.imread(psf_path)
            assert psf is not None, 'Image read falied'
            psf = cv2.cvtColor(psf, cv2.COLOR_BGR2GRAY)
            psf = psf.astype(np.float32)/np.sum(psf)  # normalized to sum=1
            self.psfs.append(psf)

    def real_data_preproc(self, img, kernel):
        H, W = img.shape[0:2]
        H1, W1 = np.int32(H/2), np.int32(W/2)
        img_warp = np.pad(
            img/1.2, ((H1, H1), (W1, W1), (0, 0)), mode='symmetric')

        return img_warp

    def __getitem__(self, idx):
        # load psf
        psfk = np.array(self.psfs[idx], dtype=np.float32)

        # load blurry data
        _imgk = np.array(self.imgs[idx], dtype=np.float32)/255
        if self.exp_mode == 'simu':
            imgk = _imgk
            gtk = np.array(self.gts[idx], dtype=np.float32)/255

        elif self.exp_mode == 'real':
            # imgk = _imgk
            # imgk = np.expand_dims(_imgk, 2).repeat(3, 2)
            imgk = self.real_data_preproc(_imgk, psfk)
            gtk = np.zeros_like(imgk, dtype=np.float32)

        return imgk.transpose(2, 0, 1).astype(np.float32), psfk[np.newaxis, :], gtk.transpose(2, 0, 1)


    def __len__(self):
        return self.img_num


# =================
# get dataloader
# =================

def get_data_loaders(img_dir=None,  patch_size=256, batch_size=8, tform_op=None, noise_type='gaussian', noise_params={'sigma': 0.05}, motion_len=[10, 20], load_psf_dir=None, load_blur_image_dir=None,  shuffle=True, validation_split=None, status='train', num_workers=8, pin_memory=False, prefetch_factor=2, all2CPU=True, test_mode='one2all', exp_mode='simu'):
    # dataset
    if status == 'train' or status == 'test' or status == 'debug' or status == 'valid':
        if all2CPU:
            raise(NotImplemented)
        else:
            raise(NotImplemented)
    elif status == 'exp':
        dataset = BlurImgDataset_Exp_all2CPU(
            load_blur_image_dir, load_psf_dir, img_dir, patch_sz=patch_size, exp_mode=exp_mode)

    loader_args = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'prefetch_factor': prefetch_factor,
        'pin_memory': pin_memory
    }

    # dataset split & dist train assignment
    if status == 'train' or status == 'debug' or status == 'valid':
        # split dataset into train and validation set
        num_total = len(dataset)
        if isinstance(validation_split, int):
            assert validation_split > 0
            assert validation_split < num_total, "validation set size is configured to be larger than entire dataset."
            num_valid = validation_split
        elif isinstance(validation_split, float):
            num_valid = int(num_total * validation_split)
        else:
            num_valid = 0  # don't split valid set

        num_train = num_total - num_valid

        train_dataset, valid_dataset = random_split(
            dataset, [num_train, num_valid])

        # distribution trainning setting
        train_sampler, valid_sampler = None, None
        if dist.is_initialized():
            loader_args['shuffle'] = False
            train_sampler = DistributedSampler(train_dataset)
            valid_sampler = DistributedSampler(valid_dataset)
        return DataLoader(train_dataset, sampler=train_sampler, **loader_args), \
            DataLoader(valid_dataset, sampler=valid_sampler, **loader_args)
    else:
        return DataLoader(dataset, **loader_args)

