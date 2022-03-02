# python train.py --epoch 15 --dataset MaskBaseByKindDataset --kind ageMod10 --model EfficientB2 --val_ratio 0.2 --name B2_AGE_FACE_CROP --data_dir /opt/ml/input/data/train/images_face_crop
# python train.py --epoch 15 --dataset MaskBaseByKindDataset --kind ageMod10 --model EfficientB2 --val_ratio 0.2 --name B2_AGE_SEG_CROP --data_dir /opt/ml/input/data/train/images_seg_crop
# python train.py --epoch 15 --dataset MaskBaseByKindDataset --kind ageMod10 --model EfficientB2 --val_ratio 0.01 --name B2_AGE_FACE_CROP --data_dir /opt/ml/input/data/train/images_face_crop
# python train.py --epoch 15 --dataset MaskBaseByKindDataset --kind ageMod10 --model EfficientB2 --val_ratio 0.01 --name B2_AGE_SEG_CROP --data_dir /opt/ml/input/data/train/images_seg_crop
# python train.py --epoch 15 --dataset MaskBaseByKindDataset --kind ageMod10 --model Resnet18 --val_ratio 0.01 --name RESNET18_AGEMOD10_FULL --data_dir /opt/ml/input/data/train/images_resize
# python train.py --epoch 15 --dataset MaskBaseByKindDataset --kind ageMod10 --model Resnet18 --val_ratio 0.01 --name RESNET18_AGEMOD10_FACE_CROP_FULL --data_dir /opt/ml/input/data/train/images_face_crop
# python train.py --epoch 15 --dataset MaskBaseByKindDataset --kind ageMod10 --model Resnet18 --val_ratio 0.01 --name RESNET18_AGEMDO10_SEG_CROP_FULL --data_dir /opt/ml/input/data/train/images_seg_crop2
# python train.py --epoch 15 --dataset MaskBaseByKindDataset --kind ageMod10 --model Resnet18 --val_ratio 0.01 --name RESNET18_AGEMDO10_SEG_CROP_DEVIDE_FULL --data_dir /opt/ml/input/data/train/images_seg_crop_devide
# python train.py --epoch 15 --dataset MaskBaseByKindDataset --model Resnet18 --val_ratio 0.02 --name RESNET18_AGE --data_dir /opt/ml/input/data/train/images
# python train.py --epoch 15 --dataset MaskBaseByKindDataset --model Resnet18 --val_ratio 0.02 --name RESNET18_AGE --data_dir /opt/ml/input/data/train/images
# python train.py --epoch 15 --dataset MaskBaseDataset --model Resnet18 --val_ratio 0.01 --name RESNET18_SEG_Crop_FULLTRAIN_ALL --data_dir /opt/ml/input/data/train/images_seg_crop2
# python train.py --epoch 15 --dataset MaskBaseDataset --model resnet34 --val_ratio 0.01 --name RESNET34_FACE_Crop_FULLTRAIN_ALL --data_dir /opt/ml/input/data/train/images_face_crop
# python train.py --epoch 15 --dataset MaskBaseDataset --model resnet34 --val_ratio 0.01 --name RESNET34_SEG_Crop_FULLTRAIN_ALL --data_dir /opt/ml/input/data/train/images_seg_crop2

#!/bin/bash
# python train.py --name RES18_AUGU_MASK --kind 'mask'
# python train.py --name RES18_AUGU_MASK0_GENDER --kind 'gender' --mask 0
# python train.py --name RES18_AUGU_MASK1_GENDER --kind 'gender' --mask 1
# python train.py --name RES18_AUGU_MASK2_GENDER --kind 'gender' --mask 2
python train.py --name RES18_MASK0_GENDER0_AGEMOD10_FC --kind 'ageMod10' --mask 0 --gender 0 --imageType 'images_face_crop'
python train.py --name RES18_MASK1_GENDER0_AGEMOD10_FC --kind 'ageMod10' --mask 1 --gender 0 --imageType 'images_face_crop'
python train.py --name RES18_MASK2_GENDER0_AGEMOD10_FC --kind 'ageMod10' --mask 2 --gender 0 --imageType 'images_face_crop'
python train.py --name RES18_MASK0_GENDER1_AGEMOD10_FC --kind 'ageMod10' --mask 0 --gender 1 --imageType 'images_face_crop'
python train.py --name RES18_MASK1_GENDER1_AGEMOD10_FC --kind 'ageMod10' --mask 1 --gender 1 --imageType 'images_face_crop'
python train.py --name RES18_MASK2_GENDER1_AGEMOD10_FC --kind 'ageMod10' --mask 2 --gender 1  --imageType 'images_face_crop'

python train.py --name RES18_MASK0_GENDER0_AGEMOD10_SC --kind 'ageMod10' --mask 0 --gender 0 --imageType 'images_seg_crop'
python train.py --name RES18_MASK1_GENDER0_AGEMOD10_SC --kind 'ageMod10' --mask 1 --gender 0 --imageType 'images_seg_crop'
python train.py --name RES18_MASK2_GENDER0_AGEMOD10_SC --kind 'ageMod10' --mask 2 --gender 0 --imageType 'images_seg_crop'
python train.py --name RES18_MASK0_GENDER1_AGEMOD10_SC --kind 'ageMod10' --mask 0 --gender 1 --imageType 'images_seg_crop'
python train.py --name RES18_MASK1_GENDER1_AGEMOD10_SC --kind 'ageMod10' --mask 1 --gender 1 --imageType 'images_seg_crop'
python train.py --name RES18_MASK2_GENDER1_AGEMOD10_SC --kind 'ageMod10' --mask 2 --gender 1  --imageType 'images_seg_crop'

python train.py --name RES18_MASK0_GENDER0_AGEMOD10_SCFU --kind 'ageMod10' --mask 0 --gender 0 --imageType 'images_seg_crop_upper_face'
python train.py --name RES18_MASK1_GENDER0_AGEMOD10_SCFU --kind 'ageMod10' --mask 1 --gender 0 --imageType 'images_seg_crop_upper_face'
python train.py --name RES18_MASK2_GENDER0_AGEMOD10_SCFU --kind 'ageMod10' --mask 2 --gender 0 --imageType 'images_seg_crop_upper_face'
python train.py --name RES18_MASK0_GENDER1_AGEMOD10_SCFU --kind 'ageMod10' --mask 0 --gender 1 --imageType 'images_seg_crop_upper_face'
python train.py --name RES18_MASK1_GENDER1_AGEMOD10_SCFU --kind 'ageMod10' --mask 1 --gender 1 --imageType 'images_seg_crop_upper_face'
python train.py --name RES18_MASK2_GENDER1_AGEMOD10_SCFU --kind 'ageMod10' --mask 2 --gender 1  --imageType 'images_seg_crop_upper_face'