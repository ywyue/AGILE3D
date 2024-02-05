#!/usr/bin/env bash

python main.py --dataset_mode=multi_obj
               --scan_folder=/cluster/work/igp_psr/yuayue/thesis/backup/reproduce/Inter3D/data/preprocess_correct \
               --train_list=/cluster/scratch/yuayue/thesis/release/AGILE3D_release/data/ScanNet/train_samples_OnlineEachScene.json \
               --val_list=/cluster/work/igp_psr/yuayue/thesis/backup/reproduce/Inter3D/data/val_scannet_randomEachScene_max10_close.jsond \
               --lr=1e-4 \
               --epochs=1100 \
               --lr_drop=[1000] \
               --job_name=train_multi_obj_scannet40