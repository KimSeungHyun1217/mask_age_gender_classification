import torch

from transform import get_default_transform
from dataloader import custom_data_loader
from model import CustomNewNet

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _transform = get_default_transform()
    mask_data_loader = custom_data_loader(kind='mask', transform=_transform, batch_size=16,
                           shuffle=True, num_workers=4)
    
    resnet = CustomNewNet(n_class=3).to(device)