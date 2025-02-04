# TAG-SPARK: TAG-lens-based SPAtial redundancy-driven noise Reduction Kernel 
Our work has been published in Advanced Science. [**paper**](<https://doi.org/10.1002/advs.202405293> "Title")

## Contents
* [Overview](#Overview)
* [Directory Structure](#DirectoryStructure)
* [Python Code](#PythonCode)
* [Results](#Results)
* [Citation](#Citation)

## Overview
Two-photon high-speed fluorescence calcium imaging stands as a mainstream technique in neuroscience for capturing neural activities with high spatiotemporal resolution. However, challenges arise from the inherent tradeoff between acquisition speed and image quality, grappling with a low signal-to-noise ratio (SNR) due to limited signal photon flux. ___Here, a contrast-enhanced video-rate volumetric system, integrating a tunable acoustic gradient (TAG) lens-based high-speed microscopy with a TAG-SPARK denoising algorithm is demonstrated.___  
The former facilitates high-speed dense z-sampling at sub-micrometer-scale intervals, allowing the latter to exploit the spatial redundancy of z-slices for self-supervised model training. ___This spatial redundancy-based approach, tailored for 4D (xyzt) dataset, not only achieves >700% SNR enhancement but also retains fast-spiking functional profiles of neuronal activities.___  
High-speed plus high-quality images are exemplified by in vivo Purkinje cells calcium observation, revealing intriguing dendritic-to-somatic signal convolution, i.e., similar dendritic signals lead to reverse somatic responses. ___This tailored technique allows for capturing neuronal activities with high SNR, thus advancing the fundamental comprehension of neuronal transduction pathways within 3D neuronal architecture.___
![advs9491-fig-0001-m](https://hackmd.io/_uploads/Sk_1iRkFJl.jpg)
![advs9491-fig-0001-m](https://hackmd.io/_uploads/H1GG7TJYkg.jpg)

## Directory Structure
:::spoiler Click to unfold the directory tree

TAG-SPARK
|---- datasets
|---- |---- noisy images (project name)
|---- |---- |---- VOL_1.tiff
|---- |---- |---- VOL_2.tiff
|---- deepcad
|---- |---- buildingblocks.py
|---- |---- dataprocess.py
|---- |---- model_3DUnet.py
|---- |---- network.py
|---- |---- test_collection.py
|---- |---- train_collection.py
|---- |---- utils.py
|---- onnx
|---- |---- model (model name)
|---- |---- |---- model.onnx
|---- pth
|---- |---- model (model name)
|---- |---- |---- model.pyh
|---- |---- |---- model.yaml
|---- results
|---- |---- denoised images (project name)
|---- |---- |---- VOL_1_denoised.tiff
|---- |---- |---- VOL_2_denoised.tiff
|---- demo_train_pipeline.py
|---- demo_test_pipeline.py

:::

## Python Code
### Updates
:pushpin: ver.1, 2023, Kai-Chun Jhan

### Our environment 
* Windows 10
* Python 3.9
* Pytorch 1.8.0
* NVIDIA GPU (GeForce RTX 3080) + CUDA (12.2)

How to install CUDA/Cudnn  
https://medium.com/geekculture/install-cuda-and-cudnn-on-windows-linux-52d1501a8805

### Setup
**1. Install CUDA**  
Ensure that CUDA is installed and properly configured for your system.  
**2. Create a Virtual Environment**  
Set up a Python virtual environment (Python ≥ 3.9).  
**3. Clone the Repository and Install Dependencies**  
Clone the repository and install the required dependencies, including PyTorch.
   ```
   $ conda create -n tagspark python=3.9
   $ conda activate tagspark
   $ pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
   $ pip install -r requirements.txt
   ```

  
### Training
A sample dataset is available in the datasets folder. Activate the virtual environment and navigate to the project directory before running `demo_train_pipeline.py`. Once completed, you can proceed with training the TAG-SPARK model. To use a custom dataset, update the `datasets_path` variable accordingly.
```
$ conda activate tagspark
$ python demo_train_pipeline.py
```

### Testing
Run `demo_test_pipeline.py` to evaluate the pre-trained model's denoising performance on the sample dataset. To test with your own data and a retrained model, update the `datasets_path` and `denoise_model` variables accordingly.
```
python demo_test_pipeline.py
```

### Param explanation
* **demo_train_pipeline.py**

```python=12
# %% Select file(s) to be processed

datasets_path = f'datasets/29_4_2_4'  # folder containing tif files for training

# %% First setup some parameters for training
n_epochs = 10               # the number of training epochs
GPU = '0'                   # the index of GPU used for computation (e.g. '0', '0,1', '0,1,2')
pth_dir = './pth'           # pth file and visualization result file path
num_workers = 0             # if you use Windows system, set this to 0.
```

* **demo_test_pipeline.py**

```python=12
# %% Select file(s) to be processed (download if not present)

datasets_path = f'datasets/29_4_2_4'  # folder containing tif files for testing
denoise_model = f'29_4_2_4_202310260959'  # A folder containing pth models to be tested

# %% First setup some parameters for testing
GPU = '0'                             # the index of GPU used for computation (e.g. '0', '0,1', '0,1,2')
num_workers = 0                       # if you use Windows system, set this to 0.
```

## Results
**1. Spatial and Temporal Characterization of the Volumetric TAG-SPARK Imaging**
![advs9491-fig-0002-m](https://hackmd.io/_uploads/Bk08gAJtJl.jpg)

**2. TAG-SPARK Facilitates Calcium Dynamics Analysis of Extensive PCs Populations**
![advs9491-fig-0003-m](https://hackmd.io/_uploads/Bydvg0yF1e.jpg)

More demo images are presented at the bottom of the [webpage](<https://doi.org/10.1002/advs.202405293> "Title") as ***Supporting Information***.

## Citation
If you use this code, please cite the companion paper where the original method appeared:

* Yin-Tzu Hsieh, Kai-Chun Jhan, Jye-Chang Lee, et al. TAG-SPARK: Empowering High-Speed Volumetric Imaging With Deep Learning and Spatial Redundancy. Advance Science. (2024). https://doi.org/10.1002/advs.202405293