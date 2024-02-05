#!/usr/bin/env bash

python eval_multi_obj.py --dataset_mode=multi_obj \
               --scan_folder=data/S3DIS/scans \
               --val_list=data/S3DIS/val_list.json \
               --output_dir=results/S3DIS_multi \
               --checkpoint=weights/checkpoint1099.pth