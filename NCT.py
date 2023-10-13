import torch
from torch.utils.data import Dataset
import pandas as pd
import imageio
import torchvision
from PIL import Image
import os
import numpy as np
import random

cls = {'ADI': 0, 'DEB': 1, 'LYM': 2, 'MUC': 3, 'MUS': 4, 'NORM': 5, 'STR': 6, 'TUM': 7}


def get_nct_dataset(dir, train_list=None, transform=None):
    if train_list == None:
        patches_dir = dir + 'patches/'
        assert os.path.exists(patches_dir)
        img_list = os.listdir(patches_dir)
    else:
        img_list = train_list
    return NCT(dir, img_list, transform)


def get_img_list(path, k=None):
    total_list = os.listdir(path)
    total_list.sort()
    random.shuffle(total_list)
    if k is None:
        return total_list
    else:
        return total_list[:k]

class NCT(Dataset):
    def __init__(self, data_dir, img_list, transform, model_type=None):
        super(NCT, self).__init__()
        self.data_dir = data_dir
        self.img_list = img_list
        self.model_type = model_type
        self.transform = transform

    def __getitem__(self, index):
        img_name = self.img_list[index].split('.')[0]
        label_name = self.img_list[index].split('-')[0]
        img = imageio.imread(os.path.join(self.data_dir + 'patches/', img_name + '.tif'))
        H_ori = imageio.imread(os.path.join(self.data_dir + 'H_RGB/', img_name + '_H.png'))
        # E_ori = imageio.imread(os.path.join(self.data_dir + 'E_RGB/', img_name + '_E.png'))
        img = Image.fromarray(img.astype('uint8'))

        # H_ori = _add_channels(H_ori)
        # E_ori = _add_channels(E_ori)

        H_ori = Image.fromarray(H_ori.astype('uint8'))
        # E_ori = Image.fromarray(E_ori.astype('uint8'))

        label = cls[label_name]

        if self.transform is not None:
            return self.transform.deal_HE(img, H_ori), label, img_name

        return img, H_ori, label, img_name

    def __len__(self):
        return len(self.img_list)
