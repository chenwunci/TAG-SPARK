# TAG-SPARK 
This repository is derived from DeepCAD-RT(https://github.com/cabooster/DeepCAD-RT)  
Other modification is in the header of the function.  
TAG-SPARK ver1. 2023  Kai-Chun Jhan  

### Our environment 

* Windows 10
* Python 3.9
* Pytorch 1.8.0
* NVIDIA GPU (GeForce RTX 3080) + CUDA (12.2)

How to install CUDA/Cudnn  
https://medium.com/geekculture/install-cuda-and-cudnn-on-windows-linux-52d1501a8805

### Setup
##### 1.Install CUDA
##### 2.Create virtual environment in Python>=3.9
##### 3.Clone repo and install dependencies(requirements.txt), including Pytorch
   ```
   $ conda create -n tagspark python=3.9
   $ conda activate tagspark
   $ pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
   $ pip install -r requirements.txt
   ```

  
### Training

There is a single sample dataset existing in the datasets folder. Please switch to the virtual environment and directory to execute `demo_train_pipeline.py`. Afterward, you can train the new TAG-SPARK model. If you wish to use your own data, please modify the `datasets_path` variable.

```
$ conda activate tagspark
$ python demo_train_pipeline.py
```

### Testing

You can run demo_test_pipeline.py to examine the pre-trained model's denoising effect on sample data. Alternatively, you can modify datasets_path and denoise_model to use your own data and a retrained model.

```
python demo_test_pipeline.py
```

### Param explanation

###### demo_train_pipeline.py

```python=11
## %% Select file(s) to be processed

datasets_path = f'datasets/29_4_2_4'  # folder containing tif files for training

# %% First setup some parameters for training
n_epochs = 10               # the number of training epochs
GPU = '0'                   # the index of GPU used for computation (e.g. '0', '0,1', '0,1,2')
pth_dir = './pth'           # pth file and visualization result file path
num_workers = 0             # if you use Windows system, set this to 0.
```


###### demo_test_pipeline.py

```python=11
# %% Select file(s) to be processed (download if not present)

datasets_path = f'datasets/29_4_2_4'  # folder containing tif files for testing
denoise_model = f'29_4_2_4_202310260959'  # A folder containing pth models to be tested

# %% First setup some parameters for testing
GPU = '0'                             # the index of GPU used for computation (e.g. '0', '0,1', '0,1,2')
num_workers = 0                       # if you use Windows system, set this to 0.
```


(reference https://github.com/cabooster/DeepCAD-RT/blob/main/README.md)
