# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import numpy as np
import torch
from einops import rearrange as rearrange


class RandomMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        np.random.shuffle(mask)
        return mask  # [196]


class RandomPatchMaskingGenerator:
    def __init__(self, mask_ratio=0.2):
        self.num_of_pixel = 16 * 16
        self.ratio = mask_ratio
        self.num_mask_pixel = int(self.num_of_pixel * self.ratio)

        self.num_of_patch = 14 * 14

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self):
        is_first = True
        for i in range(0, self.num_of_patch):
            _mask = np.hstack([
                np.zeros(self.num_mask_pixel),
                np.ones(self.num_of_pixel - self.num_mask_pixel),
            ])
            np.random.shuffle(_mask)
            _mask = torch.tensor(_mask).unsqueeze(0)
            if is_first:
                mask = _mask
                is_first = False
            else:
                mask = torch.cat((mask, _mask), dim=0)
        mask = rearrange(mask, '(n n1) (p p1) -> (n p) (n1 p1)', n=14, p=16)
        mask = mask.float()
        return mask  # [224, 224]
