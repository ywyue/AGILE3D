#!/usr/bin/env bash

python eval_single_obj.py --dataset=scannet40 \
               --dataset_mode=single_obj \
               --scan_folder=data/ScanNet/scans \
               --val_list=data/ScanNet/single/object_ids.npy \
               --val_list_classes=data/ScanNet/single/object_classes.txt \
               --output_dir=results/ScanNet_single \
               --checkpoint=weights/checkpoint1099.pth