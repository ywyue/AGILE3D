#!/usr/bin/env bash

python eval_multi_obj.py --dataset_mode=multi_obj \
               --scan_folder=data/KITTI360/scans \
               --val_list=data/KITTI360/val_list.json \
               --output_dir=results/KITTI360_multi \
               --checkpoint=weights/checkpoint1099.pth