from dataloader import custom_dataloader
from transform import get_default_transform

_transform = get_default_transform()
maskDL = custom_dataloader(kind='mask', transform=_transform, batch_size=16,
                           shuffle=True, num_workers=4)
