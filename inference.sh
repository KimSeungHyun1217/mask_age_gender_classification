# python inference.py --model EfficientB2 --kind mask --data_folder images_seg_crop --model_dir ./model/B2_SEG_Crop_MASK_FULLTRAIN --file_name B2_SC_MASK
# python inference.py --model EfficientB2 --kind gender --data_folder images_seg_crop --model_dir ./model/B2_SEG_Crop_GENDER_FULLTRAIN --file_name B2_SC_GENDER
python inference.py --data_folder images_face_crop --file_name R18_FC_AGE_AUG.csv --dtype FC
python inference.py --data_folder images_seg_crop --file_name R18_SC_AGE_AUG.csv --dtype SC
python inference.py --data_folder images_seg_crop_face_upper --file_name R18_SCFU_AGE_AUG.csv --dtype SCFU
