import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from PIL import Image

class MasedFaceDataset(Dataset):
    def __init__(self,  kind, transform):
        self.transform = transform
        self.kind = kind # mask, gender, age

        self.csv_path = '../input/data/train/mask.csv'
        self.df = pd.read_csv(self.csv_path)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path'])
        label = row[self.kind]
        img = self.transform(image=np.array(img))['image']

        return img, label
        
    def __len__(self):
        return len(self.df)

def custom_data_loader(kind, transform, batch_size=1, shuffle=False, sampler=None, num_workers=0):
    data_set = MasedFaceDataset(kind, transform)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers)
    return data_loader