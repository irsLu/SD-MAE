import torch
from PIL import Image

from torch.utils.data import Dataset
import imageio


class Camelyon16(Dataset):
    def __init__(self, root_dir, labels_path, is_train, transform=None):
        self.root_dir = root_dir
        self.img_paths = []
        self.labels = []
        self.transform = transform
        self.is_train = is_train
        self.label2cls = {}
        self.cls2label = {}
        i = 0
        with open(labels_path, 'r') as f:
            all = f.readlines()
            for each in all:
                file, img = str.split(each, '\t')
                img = str.split(img, '\n')[0] if img.endswith('\n') else img
                img_path = file + '/' + img
                label = file.split('_')[1].lower()
                if label not in self.label2cls.keys():
                    self.label2cls[label] = i
                    self.cls2label[i] = label
                    i += 1

                self.img_paths.append(img_path)
                self.labels.append(label)

    def __getitem__(self, idx):
        img_path = self.root_dir + self.img_paths[idx]
        img = imageio.imread(img_path)

        sample = Image.fromarray(img.astype('uint8'))
        if self.transform is not None:
            sample = self.transform(sample)
        target = self.label2cls[self.labels[idx]]

        return sample, target

    def __len__(self):
        return len(self.img_paths)
