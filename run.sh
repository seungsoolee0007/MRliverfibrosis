
python3 main.py --mode=train --train_data=/mnt/SSD_BIG2/LSS_LIV/CT_PNG_TRAIN/tr \
 --val_data=/mnt/SSD_BIG2/LSS_LIV/CT_PNG_TRAIN/te \
--ckpt=/mnt/SSD_BIG2/LSS_LIV/DenseNetCkpt_outliered/ --layers_per_block=4,5,6,6,7,8 \
--batch_size=2 --epochs=15 --growth_k=16 --num_classes=2 --learning_rate=0.001
