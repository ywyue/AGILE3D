#!/usr/bin/env bash

python eval_single_obj.py --dataset=kitti360 \
               --dataset_mode=single_obj \
               --scan_folder=data/KITTI360/single/crops \
               --crop \
               --val_list=data/KITTI360/single/object_ids.npy \
               --val_list_classes=data/KITTI360/single/object_classes.txt \
               --output_dir=results/KITTI_single \
               --checkpoint=weights/checkpoint1099.pth
