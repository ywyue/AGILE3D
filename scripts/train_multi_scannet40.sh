#!/usr/bin/env bash

python main.py --dataset_mode=multi_obj
               --scan_folder=data/ScanNet/scans \
               --train_list=data/ScanNet/train_list.json \
               --val_list=data/ScanNet/val_list.json \
               --lr=1e-4 \
               --epochs=1100 \
               --lr_drop=[1000] \
               --job_name=train_multi_obj_scannet40