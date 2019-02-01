
python3 main.py --mode=infer \
 --infer_data=/mnt/SSD_BIG2/LSS_LIV/CT_PNG_TRAIN/test/images \
 --output_folder=/mnt/SSD_BIG2/LSS_LIV/CT_PNG_TRAIN/infer_result_test_deeplab \
 --ckpt=/mnt/SSD_BIG2/LSS_LIV/DenseNetCkpt_second/-99 --layers_per_block=4,5,6,6,7,8 \
 --batch_size=1 
