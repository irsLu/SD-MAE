import argparse
import os.path as osp
from collections import OrderedDict

import mmcv
import torch
from mmcv.runner import CheckpointLoader


def convert_beit(ckpt):
    new_ckpt = OrderedDict()

    for k, v in ckpt.items():
        if k.startswith('encoder.'):
            new_key = k[8:]
            if new_key != 'norm.weight' and new_key != 'norm.bias':
                new_ckpt['backbone.'+new_key] = ckpt[k]
    for name, param in new_ckpt.items():
        print(name)
    return new_ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in official pretrained beit models to'
                    'MMSegmentation style.')
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()
    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
   # state_dict = state_dict['encoder']
    # for name, param in state_dict.items():
    #     print(name)
    weight = convert_beit(state_dict)
    mmcv.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)


if __name__ == '__main__':
    main()