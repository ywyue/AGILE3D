#!/usr/bin/env bash

python eval_multi_obj.py --dataset_mode=multi_obj \
               --scan_folder=data/ScanNet/scans \
               --val_list=data/ScanNet/val_list.json \
               --output_dir=results/ScanNet_multi \
               --checkpoint=weights/checkpoint1099.pth