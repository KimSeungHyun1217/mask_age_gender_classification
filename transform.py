import albumentations
from albumentations.pytorch.transforms import ToTensorV2

def get_default_transform():
    transforms = albumentations.Compose([
        ToTensorV2()
    ]);
    return transforms