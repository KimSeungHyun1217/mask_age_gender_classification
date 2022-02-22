import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim import lr_scheduler
from transform import get_default_transform
from dataloader import custom_data_loader
from model import CustomNewNet
from train import train_or_eval

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trans = get_default_transform()

    EPOCH = 20
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3

    train_mask_data_loader = custom_data_loader(csv_path='../input/data/train/mask.csv', kind='mask', 
                                          transform=trans, train=True, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=4)
    custom_model = CustomNewNet(n_class=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizaer = optim.Adam(custom_model.parameters(), lr=LEARNING_RATE)
    scheduler = lr_scheduler.StepLR(optimizaer, step_size=7, gamma=0.1)

    result_model = train_or_eval(mode='train', model=custom_model, 
                                data_loader=train_mask_data_loader, 
                                criterion=criterion, optimizer=optimizaer,
                                scheduler=scheduler, n_epoch=EPOCH, device=device)
    
    print('Done')