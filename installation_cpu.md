# Installation Instruction (CPU-only version)

We recommend setting up the environment using Miniconda or Anaconda. We have tested the code on Linux with Python 3.10, torch 1.12.1, but it should also work in other environments. If you have trouble, feel free to open an issue.

### Step 1: create an environment
Clone this repo:
```shell
git clone https://github.com/ywyue/AGILE3D.git
cd AGILE3D
```

```shell
conda create -n agile3d python=3.10
conda activate agile3d
```
### Step 2: install pytorch
```shell
pip install torch==1.12.1 torchvision==0.13.1 --index-url https://download.pytorch.org/whl/cpu
```
### Step 3: install Minkowski
3.1 Prepare:
```shell
conda install -c intel mkl mkl-include
```
3.2 Install:
```shell
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --blas=mkl --cpu_only 
```
If you run into issues, please refer to MinkowskiEngine's [official instructions for CPU only compilation](https://nvidia.github.io/MinkowskiEngine/quick_start.html#cpu-only-compilation).
### Step 4: install other packages
```shell
pip install open3d
```
