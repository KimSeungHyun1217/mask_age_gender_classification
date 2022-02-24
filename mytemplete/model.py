import torch.nn as nn

from torchvision import models

class CustomNewNet(nn.Module):
    def __init__(self, n_class):
        super(CustomNewNet, self).__init__()
        self.resnet34 = models.resnet34(pretrained=True)
        num_ftrs = self.resnet34.fc.in_features
        self.resnet34.fc = nn.Linear(num_ftrs, n_class)
    
    def forward(self, x):
        return self.resnet34(x)

