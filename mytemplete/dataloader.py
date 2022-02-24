import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from PIL import Image

class MasedFaceDataset(Dataset):
    def __init__(self, csv_path, kind=None, transform=None, train=True):
        self.kind = kind # mask, gender, age
        self.csv_path = csv_path
        self.transform = transform
        self.train = train
        self.df = pd.read_csv(self.csv_path)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if self.train:
            img = Image.open(row['path'])
            label = row[self.kind]
            if self.transform:
                img = self.transform(image=np.array(img))['image']
        else:
            img = Image.open(row['ImageID'])
            label = row['ans']
        return img, label

    def __len__(self):
        return len(self.df)

def custom_data_loader(csv_path, kind=None, transform=None, train=True, batch_size=1, shuffle=False, sampler=None, num_workers=2):
    data_set = MasedFaceDataset(csv_path, kind, transform, train)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers)
    return data_loader