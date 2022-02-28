# python inference.py --model EfficientB2 --kind mask --data_folder images_seg_crop --model_dir ./model/B2_SEG_Crop_MASK_FULLTRAIN --file_name B2_SC_MASK
# python inference.py --model EfficientB2 --kind gender --data_folder images_seg_crop --model_dir ./model/B2_SEG_Crop_GENDER_FULLTRAIN --file_name B2_SC_GENDER
python inference.py --model Resnet18 --kind ageMod10 --data_folder images_resize --model_dir ./model/RESNET18_AGEMOD10_FULL --file_name R18_AGE.csv
python inference.py --model Resnet18 --kind ageMod10 --data_folder images_face_crop --model_dir ./model/RESNET18_AGEMOD10_FACE_CROP_FULL --file_name R18_FC_AGE.csv
python inference.py --model Resnet18 --kind ageMod10 --data_folder images_seg_crop --model_dir ./model/RESNET18_AGEMOD10_SEG_CROP_FULL --file_name R18_SC_AGE.csv
python inference.py --model Resnet18 --kind ageMod10 --data_folder images_seg_crop_devide --model_dir ./model/RESNET18_AGEMOD10_SEG_CROP_DEVIDE_FULL --file_name R18_SCD_AGE.csv
