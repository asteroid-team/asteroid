#!/bin/bash

# Exit on error
set -e
set -o pipefail

# Main storage directory
storage_dir=

# If you want to download the development dataset, specify which dataset you want to download
# Set both variable to true if you want to download the dry and reverberated datasets 
# and start from stage 0
fuss_dry=true
fuss_reverb=false

# If you want to create the dataset from scratch or perform data augmentation
# set the following variables the path to the root directory and start from stage 1
random_seed=
num_train= 
num_val=

# If you already have FUSS wav files, specify the path to the directory here
# and start from stage 3
fuss_dir=

# Path to the python you'll use for the experiment. Defaults to the current python
# You can run ./utils/prepare_python_env.sh to create a suitable python environment, paste the output here.
python_path=python

# Example usage
# ./run.sh --stage 3 --tag my_tag --task sep_noisy --id 0,1

# General
stage=0  # Controls from which stage to start
tag=""  # Controls the directory name associated to the experiment
# You can ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES)
id=$CUDA_VISIBLE_DEVICES


# Data
sample_rate=16000

# Training

# Architecture

# Evaluation


. utils/parse_options.sh

# Work with development data
if [ $stage -eq 0 ]; then
    echo "Stage 0: Get development data"
    bash ./local/get_dev_data.sh $storage_dir $fuss_dry $fuss_reverb
fi

# Data augmentation stage 
if [ $stage -eq 1 ]; then
    echo "Stage 1: Run data augmentation"
    bash ./local/data_augmentation.sh $storage_dir $num_train $num_val
fi

# Following stages
if [ $stage -eq 3 ]; then
    echo "Proceed with following steps"
fi
