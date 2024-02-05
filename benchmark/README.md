# Benchmark

### *[Under construction]*

This document provides more information on the benchmark setup, data preprocessing, etc.	


We provide the preprocessed data, here are some explanations for folders and files in each dataset:

- **scans**: 3D point clouds, containing 'x', 'y', 'z', 'R', 'G', 'B', 'label' attributes. 'label' is instance id, numbered from 1, 2, 3, ... For ScanNet, unlabeled points are assigned to -1.
- **train_list.json**: contains scene ids for training in multi-object setup.
- **val_list.json**: contains scene ids and selected object ids for evaluation in multi-object setup.
- **single**: contains data used for evaluation in single-object setup
	- **crops**: in interactive single object segmentation, the evaluation typically is conducted on a cropped scan centered on the target object. If we evaluate in crop mode, then this folder contains the cropped point cloud for each object.
	- **object_ids.txt**: contains object ids
	- **object_classes.txt**: contain the semantic label for each object




## Data preprocessing

The following steps are not required since we provided the preprocessed data. Nevertheless, here is the instruction if you want to prepare the data yourself.

### Download the original dataset

- **ScanNet**: Download ScanNet v2 data from the [official ScanNet website](https://github.com/ScanNet/ScanNet). You need to download as least `***_vh_clean_2.ply`, `***.aggregation.json`, `***_vh_clean_2.0.010000.segs.json` for all scenes in training and validation set.
- **S3DIS**: Download S3DIS data from the [official S3DIS website](http://buildingparser.stanford.edu/dataset.html#Download). We use the aligned version.
- **KITTI-360**: Download KITTI-360 data from the [official KITTI-360 website](https://www.cvlibs.net/datasets/kitti-360/).


### Run scripts

Coming soon.