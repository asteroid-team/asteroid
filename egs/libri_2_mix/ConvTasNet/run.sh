#!/bin/bash
set -e  # Exit on error

# Main storage directory. You'll need disk space to dump the raw LibriSpeech flac files
# and generate the mixtures
storage_dir=

# If you haven't downloaded and extracted the LibriSpeech dataset start from stage 1

# If you already have dowloaded and extracted LibriSpeech,
# specify the path to the directory here and start from stage 2
librimix_root_path=

# After running the recipe a first time, you can run it from stage 3 directly to train new models.

# Path to the python you'll use for the experiment. Defaults to the current python
# You can run ./utils/prepare_python_env.sh to create a suitable python environment, paste the output here.
#python_path=${storage_dir}/asteroid_conda/miniconda3/bin/python
python_path=python

# Example usage
# ./run.sh --stage 3 --tag my_tag --task sep_noisy --id 0,1

. utils/parse_options.sh

storage_dir=$storage_dir/libri

# General
stage=2  # Controls from which stage to start
#tag=""  # Controls the directory name associated to the experiment
## You can ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES)
#id=0,1,2,3
#
## Data
#data_dir=data  # Local data directory (No disk space needed)
#task=sep_clean  # Specify the task here (sep_clean, sep_noisy, enh_single, enh_both)
#sample_rate=16000
#
#
## Training
#batch_size=8
#num_workers=8
##optimizer=adam
#lr=0.001
#epochs=200
#
## Architecture
#n_blocks=6
#n_repeats=2
#mask_nonlinear=relu
#
## Evaluation
#eval_use_gpu=1
exp_dir=exp/tanh
test_dir=/home/jcosentino/libri/libri2mix/8K/min/metadata/mixture_test-clean.csv

if [[ $stage -le  -1 ]]; then
	echo "Stage -1: Creating python environment to run this"
	if [[ -x "${python_path}" ]]
	then
		echo "The provided python path is executable, don't proceed to installation."
	else
	  . utils/prepare_python_env.sh --install_dir $python_path --asteroid_root ../../..
	  echo "Miniconda3 install can be found at $python_path"
	  python_path=${python_path}/miniconda3/bin/python
	  echo -e "\n To use this python version for the next experiments, change"
	  echo -e "python_path=$python_path at the beginning of the file \n"
	fi
fi



if [[ $stage -le  1 ]]; then
	echo "Stage 1: Downloading LibriSpeech"
  . local/prepare_data.sh --storage_dir $storage_dir
fi

if [[ $stage -le  2 ]]; then
	echo "Stage 2: Generating metadata "
  $python_path local/create_metadata.py --storage_dir $storage_dir --n_src 2
fi

if [[ $stage -le  3 ]]; then
	echo "Stage 3: Generating Librimix dataset"
  $python_path local/create_dataset_from_metadata.py --librispeech_root_path $storage_dir/LibriSpeech --dataset_root_path $storage_dir/libri2mix
fi

## Generate a random ID for the run if no tag is specified
#uuid=$($python_path -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
#if [[ -z ${tag} ]]; then
#	tag=${task}_${sr_string}k${mode}_${uuid}
#fi
#expdir=exp/train_convtasnet_${tag}
#mkdir -p $expdir && echo $uuid >> $expdir/run_uuid.txt
#echo "Results from the following experiment will be stored in $expdir"


if [[ $stage -le 4 ]]; then
  echo "Stage 4: Training"
  mkdir -p logs
#  CUDA_VISIBLE_DEVICES=$id $python_path train.py --exp_dir exp/8K_mss \
#  --exp_dir ${expdir}/ | tee logs/train_${tag}.log
  $python_path train.py --exp_dir $exp_dir
fi


if [[ $stage -le 5 ]]; then
	echo "Stage 5 : Evaluation"
#	CUDA_VISIBLE_DEVICES=$id $python_path eval.py
  $python_path eval.py --exp_dir $exp_dir --test_dir $test_dir
fi
