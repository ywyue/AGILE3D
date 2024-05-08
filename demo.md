# Interactive 3D segmentation tool

<p align="center">
<img src="./imgs/demo.gif" width="75%" />
</p>

We present an **interactive tool** that allows users to segment/annotate multiple 3D objects together, in an **open-world** setting. Although our work focuses on multi-object cases, this tool can also support interactive single-object segmentation seamlessly. We provide several sample scans below but you are highly encouraged to try your own scans! 

Before you start:

- If GPU is available, we assume you already set up the environments properly following [installation.md](https://github.com/ywyue/AGILE3D/tree/main/installation.md).
- If only CPU is available, please set up the environments following [installation_cpu.md](https://github.com/ywyue/AGILE3D/tree/main/installation_cpu.md).

### Step 1: download pretained model
Download the [**model**](https://polybox.ethz.ch/index.php/s/RnB1o8X7g1jL0lM) and move it to the ```weights``` folder.

The model was only trained on [ScanNet40](http://www.scan-net.org/) training set, but it can also be used to segment scenes from other datasets, e.g., [S3DIS](http://buildingparser.stanford.edu/dataset.html), [ARKitScenes](https://github.com/apple/ARKitScenes) and even outdoor scans, [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/).

### Step 2: download sample data
[**Sample data link**](https://polybox.ethz.ch/index.php/s/HMhuyJwJkPXxP3f)

The data should be organized as follows:
```
code_root/
└── data/
    └── interactive_dataset/
        ├── scene_*****/
        |    ├── scan.ply
        |    └── label.ply (optional)
        ├── scene_*****/
        ├── ...
        └── scene_*****/

```
Note:
- ```scan.ply```: the 3D scan file, which can be a mesh or a point cloud file.
- ```label.ply``` (optional): the label file which should contain a 'label' attribute that indicates the instance id (starting from 1, 2, 3 ...) of each point. This file is optional. If provided, the system will automatically record the segmentation IoU.

Here we provide some samples from ScanNet, S3DIS, KITTI-360 (with label) and samples from ARKitScenes (without label). **You may also want to try your own scans!** Note for large-scale scans, you may need to crop them into a smaller part in case out of memory.

### Step 3: run the tool
Run the UI using the following command:
```shell
python run_UI.py --user_name=test_user --pretraining_weights=weights/checkpoint1099.pth --dataset_scenes=data/interactive_dataset
```
<details>
<summary><span style="font-weight: bold;">Important Command Line Arguments for run_ui.py</span></summary>
    
  #### --user_name
  User name, useful for user study.
  #### --pretraining_weights
  Path where the trained model was stored.
  #### --point_type
  'mesh' or 'pointcloud'. If not given, the type will be determined automatically.
    
</details>

### Quick user instructions:
- Basics for 3D manipulation: Long press the left mouse button: rotation; Long press the right mouse button: translation.
- **Numpad keys (starting from 1) + left click** will identify one **object**, **ctrl + left click** will identify the **background**. The key should be held down before the left clicking.
- In the first round, please identify all interested objects - one click per object.
- After the first round, correct errors. Again, Numpad keys i + left click will assign the desired part to **object i** and ctrl + left click will assign the desired part to **background**.
- The user can tick the checkbox **"Infer automatically when clicking"**, then the model will automatically return a new segmentation mask after each click, so no need to click the button **"RUN/SAVE [Enter]"**.
- The above instructions are not deterministic - you can also segment objects one by one.

### Instructions for your own scans:
- Please make sure the **z-axis** is up.
- The scan should be at metric scale, i.e. the point coordinates should be in meters. If the point coordinates are too large, it is better to move them to (0,0,0) as the origin.
- The ```scan.ply``` should contain as least **'x', 'y', 'z', 'red', 'green', 'blue'** attributes. If your scan does not have color information, please set 'red', 'green', 'blue' as 0.
- If the scan is too large, please crop a small part.