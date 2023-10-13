# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import os
import torch
import PIL
from torchvision import datasets, transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

from timm.data import create_transform
from masking_generator import RandomMaskingGenerator
from dataset_folder import ImageFolder
from Camelyon16 import Camelyon16
from MHIST import MHIST

from NCT import get_nct_dataset


class DataAugmentationForMAE(object):
    def __init__(self, args):
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

        if args.withRRC:
            if args.min_scale > 0:
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(args.input_size, (args.min_scale, 1), ratio=(1, 1),
                                                 interpolation=PIL.Image.BICUBIC),
                    # transforms.RandomResizedCrop(args.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=torch.tensor(mean),
                        std=torch.tensor(std))
                ])
            else:
                self.transform = transforms.Compose([
                    # transforms.RandomResizedCrop(args.input_size, (args.min_scale, 1), ratio=(1,1),interpolation=PIL.Image.BICUBIC),
                    transforms.RandomResizedCrop(args.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=torch.tensor(mean),
                        std=torch.tensor(std))
                ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize([args.input_size, args.input_size]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=torch.tensor(mean),
                    std=torch.tensor(std))
            ])

        self.masked_position_generator = RandomMaskingGenerator(
            args.window_size, args.mask_ratio
        )

    def __call__(self, image):
        return self.transform(image), self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForBEiT,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


def build_pretraining_dataset(args):
    transform = DataAugmentationForMAE(args)
    print("Data Aug = %s" % str(transform))

    if args.data_set == "camelyon":
        labels_path = os.path.join(args.data_path,
                                   'camelyon16_train_labels.txt')
        root_dir = os.path.join(args.data_path, 'camelyon16/')
        dataset = Camelyon16(root_dir=root_dir, labels_path=labels_path, is_train=True, transform=transform)
    elif args.data_set == "camelyon0.1":
        labels_path = os.path.join(args.data_path,
                                   'camelyon16_train_labels_0.1.txt')
        root_dir = os.path.join(args.data_path, 'camelyon16/')
        dataset = Camelyon16(root_dir=root_dir, labels_path=labels_path, is_train=True, transform=transform)
    elif args.data_set == "ImageNet100":
        dataset = ImageFolder(args.data_path, transform=transform)
    elif args.data_set == 'nct':
        dataset = get_nct_dataset(args.data_path, transform=transform)
    elif args.data_set == 'MHIST':
        root_dir = os.path.join(args.data_path, 'MHIST/images/')
        labels_path = os.path.join(args.data_path, 'MHIST/annotations.csv')
        dataset = MHIST(root_dir=root_dir, labels_path=labels_path, is_train=True, transform=transform)

    return dataset


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if (not is_train) and args.eval:
        root = os.path.join(args.data_path, 'test')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 2
    elif args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'Default':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'ImageNet100':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 100
    elif args.data_set == 'nct':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 8
    elif args.data_set == 'nct_4':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 4
    elif args.data_set == 'nct_2':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 2
    elif args.data_set == 'chaoyang':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 4
    elif args.data_set == "camelyon":
        labels_path = os.path.join(args.data_path,
                                   'camelyon16_train_labels.txt' if is_train else 'camelyon16_val_labels.txt')
        root_dir = os.path.join(args.data_path, 'camelyon16/')
        dataset = Camelyon16(root_dir=root_dir, labels_path=labels_path, is_train=True, transform=transform)
        nb_classes = 2
    elif args.data_set == "camelyon0.1":
        labels_path = os.path.join(args.data_path,
                                   'camelyon16_train_labels_0.1.txt' if is_train else 'camelyon16_val_labels_0.1.txt')
        root_dir = os.path.join(args.data_path, 'camelyon16/')
        dataset = Camelyon16(root_dir=root_dir, labels_path=labels_path, is_train=True, transform=transform)
        nb_classes = 2
    elif args.data_set == "image_folder":
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    elif args.data_set == 'MHIST' or args.data_set == 'pCam':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 2
    elif args.data_set == 'covid':
        dataset = datasets.ImageFolder(args.data_path, transform=transform)
        t = int(len(dataset) * 0.8)
        v = len(dataset) - t
        train, val = torch.utils.data.random_split(dataset, [t, v])
        return train, val, 4
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)
    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        if args.crop_pct is None:
            if args.input_size < 384:
                args.crop_pct = 224 / 256
            else:
                args.crop_pct = 1.0
        size = int(args.input_size / args.crop_pct)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
