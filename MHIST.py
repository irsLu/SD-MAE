from torch.utils.data import Dataset
import pandas as pd
import imageio
from PIL import Image
import os


class MHIST(Dataset):
    def __init__(self, labels_path, root_dir, transform, is_train):
        super(MHIST, self).__init__()
        df = pd.read_csv(labels_path)
        self.cls2index = {'SSA': 0, 'HP': 1}
        self.root_dir = root_dir
        self.transform = transform
        if is_train:
            self.img_paths = df[df['Partition'] == 'train']
        else:
            self.img_paths = df[df['Partition'] == 'test']

    def __getitem__(self, idx):
        # print(z.iloc[0]['Image Name'])
        img_path = os.path.join(self.root_dir, self.img_paths.iloc[idx]["Image Name"])
        label = self.cls2index[self.img_paths.iloc[idx]["Majority Vote Label"]]
        img = imageio.imread(img_path)
        sample = Image.fromarray(img.astype('uint8'))
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label

    def __len__(self):
        return len(self.img_paths)
