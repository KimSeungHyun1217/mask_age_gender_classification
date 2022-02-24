import torch
import torch.nn as nn
import torch.optim as optim
import os

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
    criterion = nn.CrossEntropyLoss()

    mask_data_loader = custom_data_loader(csv_path='../input/data/train/mask.csv', kind='mask', 
                                          transform=trans, train=True, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=4)
    mask_model = CustomNewNet(n_class=3).to(device)
    mask_optimizaer = optim.Adam(mask_model.parameters(), lr=LEARNING_RATE)
    mask_scheduler = lr_scheduler.StepLR(mask_optimizaer, step_size=7, gamma=0.1)

    age_data_loader = custom_data_loader(csv_path='../input/data/train/mask.csv', kind='age', 
                                    transform=trans, train=True, batch_size=BATCH_SIZE,
                                    shuffle=True, num_workers=4)
    age_model = CustomNewNet(n_class=3).to(device)
    age_optimizaer = optim.Adam(age_model.parameters(), lr=LEARNING_RATE)
    age_scheduler = lr_scheduler.StepLR(age_optimizaer, step_size=7, gamma=0.1)


    print('-' * 30)
    print('mask train start')
    result_mask_model = train_or_eval(mode='train', model=mask_model, 
                                data_loader=mask_data_loader, 
                                criterion=criterion, optimizer=mask_optimizaer,
                                scheduler=mask_scheduler, n_epoch=EPOCH, device=device)   
    print('-' * 30)
    print('age train start')
    result_age_model = train_or_eval(mode='train', model=age_model, 
                                data_loader=age_data_loader, 
                                criterion=criterion, optimizer=age_optimizaer,
                                scheduler=age_scheduler, n_epoch=EPOCH, device=device)


    
    save_path = '../model'
    torch.save(result_mask_model.state_dict(), os.path.join(save_path, "mask_model.pt")) 
    print('-' * 30)
    print('Save mask Model to {}'.format(save_path))
    print('-' * 30)
    torch.save(result_mask_model.state_dict(), os.path.join(save_path, "age_model.pt"))
    print('Save age Model to {}'.format(save_path))
    print('-' * 30)