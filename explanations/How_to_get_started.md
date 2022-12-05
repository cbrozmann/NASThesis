# Prerequisite
## Install CUDA
If you have an NVIDIA GPU please download and install [CUDA drivers](https://developer.nvidia.com/cuda-downloads) like described from NVIDIA.

## Python
We used <b>Miniconda3</b>.
Inside we created a python 3.8 distribution.
    
    conda create -n <user_name> python=3.8
    conda activate <user_name>

## Pytorch and Torch Home
Install [pytorch](https://pytorch.org/get-started/locally/), if you use conda f.e. with the following command

    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

And set a TORCH_HOME global variable with a path to the directory where all architectures and data sets should be stored.
On linux you can use 

    export TORCH_HOME=<path>/TORCH_HOME

## Architectures
For the installation of the architectures install the archive (tss) from <https://github.com/D-X-Y/NATS-Bench>.
Extract the folder NATS-tss-v1_0-3ffb9-simple into the TORCH_HOME directory.

## XAutoDL
Optionally create a git directory/folder where all gits for this thesis can be saved.

    mkdir git
    cd git/

Download XAutoDL (Program to reconstruct NAS-Bench-201 and NATS-Bench or train the architectures of the search space from them) with the command

    git clone --recurse-submodules https://github.com/D-X-Y/AutoDL-Projects.git XAutoDL
    
## Install NAS-Bench-201
Run the command to install nas-bench-201

    pip install nas-bench-201

# Installation and Preparation of the thesis repository
## Download
Download the repository ideally in the git folder from above
    
    git clone https://gitlab.uni-koblenz.de/cbrozmann/nas-thesis.git nas-thesis

## Data sets to TORCH_HOME
You can use one of two options:
1. Move the data sets from the thesis repository to the TORCH_HOME directory
2. Alternatively create the data set from the

In my case the instructions for the first option are

    mv git/nas-thesis/data_sets/UTD-MHAD TORCH_HOME
    mv git/nas-thesis/data_sets/ntu_60 TORCH_HOME

for the second option
### Transform UTD-MHAD data set
1. Download the skeleton data from the [UTD-MHAD data set](http://www.utdallas.edu/~kehtar/UTD-MAD/Skeleton.zip)
2. Extract the data set/unzip the file
3. Transform the skeleton data to images (.png). 
   1. Change the input and output folders in the file *nas-thesis/code/utd_mhad_transform_skeleton_data_to_images.py* and run the file. 
      1. Change the input folder (*input_path*) to the path the data set was extracted
      2. Change the output folder (*output_root_path*) to *<TORCH_HOME_path>/UTD-MHAD*.
      3. Run the file

### Transform NTU RGB+D data set
1. Download the skeleton data from the [NTU RGB+D data set](https://rose1.ntu.edu.sg/dataset/actionRecognition/) (nturgbd_skeletons_s001_to_s017.zip).
Alternatively: While the thesis was written the same file was linked on the authors [GitHub](https://github.com/shahroudy/NTURGB-D)
2. Extract the data set/unzip the file
3. Transform the .mat files to .npy files like described on the authors [GitHub](https://github.com/shahroudy/NTURGB-D/tree/master/Python)
   1. Download the file with the missing samples. [NTU_RGBD120_samples_with_missing_skeletons.txt](https://github.com/shahroudy/NTURGB-D/blob/master/Matlab/NTU_RGBD120_samples_with_missing_skeletons.txt) or [NTU_RGBD_samples_with_missing_skeletons.txt](https://github.com/shahroudy/NTURGB-D/blob/master/Matlab/NTU_RGBD_samples_with_missing_skeletons.txt)
   2. Download the file [txt2npy.py](https://github.com/shahroudy/NTURGB-D/blob/master/Python/txt2npy.py)
   3. Change and run the file *txt2npy.py*
      1. Change the value of the variable *save_npy_path* with the path you want to save the .npy files (you will need path to this folder later)
      2. Change the value of the variable *load_txt_path* with the path you extracted the data set data (should look like "*path_you_downloaded/nturgb+d_skeletons/*").
      3. Change the value of the variable *missing_file_path* with the path you downloaded the file with the missing samples.
      4. run the file f.e. with the shell command ```python txt2npy.py```
4. Transform the skeleton data from ".npy" to images (.png).
   1. Change the input and output folders in the file *nas-thesis/code/utd_mhad_transform_skeleton_data_to_images.py* and run the file. 
      1. Change the input folder (*input_path*) to the path the data set was extracted
      2. Change the output folder (*output_root_path*) to *<TORCH_HOME_path>/ntu_60*
      3. Run the file

## Move files from thesis repository to XAutoDL repository
Move all files from the folder *nas-thesis/changed_code/* to the corresponding folder in the XAutoDL repository.
If you have both repositories in the same folder you can use the following commands in this folder:

    cp nas-thesis/changed_code/configs/nas-benchmark/*.json XAutoDL/configs/nas-benchmark/
    cp nas-thesis/changed_code/configs/nas-benchmark/hyper-opts/* XAutoDL/configs/nas-benchmark/hyper-opts/
    cp nas-thesis/changed_code/scripts/NATS-Bench/* XAutoDL/scripts/NATS-Bench/
    cp nas-thesis/changed_code/xautodl/datasets/* XAutoDL/xautodl/datasets/
    cp nas-thesis/changed_code/exps/NATS-Bench/* XAutoDL/exps/NATS-Bench/


## Install changed XAutoDL repository
Run the following command in the folder of the XAutoDL repository:

    pip install .

# Train the architectures
Open the XAutoDL directory. The following command trains all architectures with 200 Epochs with three different seeds on the UTD-MHAD data set.

    bash scripts/NATS-Bench/own_train-topology_no_augmentation.sh 00000-15624 200 "777 888 999"


Similar for the cross subject and cross view splits of the NTU RGB+D data set

    bash scripts/NATS-Bench/own_train-ntu60_cross_subject.sh 00000-15624 200 "777 888 999"
    bash scripts/NATS-Bench/own_train-ntu60_cross_view.sh 00000-15624 200 "777 888 999"


the commands have the following structure

    bash scripts/NATS-Bench/<script> <Start Architecture>-<End Architecture> <Number Epochs (1,12,200)> "<Seeds>"

you can run these commands on specific NVIDIA GPUS with prefixing *CUDA_VISIBLE_DEVICES=<GPU_ID>*
e.g.

    CUDA_VISIBLE_DEVICES=0 bash scripts/NATS-Bench/own_train-topology_no_augmentation.sh 00000-15624 200 "777 888 999"



    



