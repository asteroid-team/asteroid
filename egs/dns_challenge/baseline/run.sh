#!/bin/bash

# Exit on error
set -e
set -o pipefail

# Main storage directory. You'll need disk space to dump DNS dataset.
storage_dir=
# If you want to clone the DNS-Challenge repo somewhere different
clone_dir=  # optional

# Path to the python you'll use for the experiment. Defaults to the current python
# You can run ./utils/prepare_python_env.sh to create a suitable python environment.
python_path=python

# Example usage
# ./run.sh --stage 3 --tag my_tag --is_complex true --id 0,1

# General
stage=0
tag=""  # Controls the directory name associated to the experiment
# You can ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES)
id=$CUDA_VISIBLE_DEVICES

# Experiment config
is_complex=1  # If we use a complex network for training.

# Evaluation
eval_use_gpu=1

. ./utils/parse_options.sh

dumpdir=data

recipe_dir=$PWD
if [[ -z ${clone_dir} ]]; then
	clone_dir=$storage_dir
fi

if [[ $stage -le  0 ]]; then
  echo "Stage 0 : Install git-lfs and download the data (this will take a while)"
  . ./local/install_git_lfs.sh
	. ./local/download_data.sh $clone_dir
fi

if [[ $stage -le  1 ]]; then
  echo "Stage 1 : Create the dataset"
  . ./local/create_dns_dataset.sh $clone_dir $storage_dir
  cd $recipe_dir
fi


if [[ $stage -le  2 ]]; then
  echo "Stage 2 : preprocess the dataset"
  python local/preprocess_dns.py --data_dir $storage_dir --json_dir $dumpdir
fi

# Generate a random ID for the run if no tag is specified
uuid=$(python -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
if [[ -z ${tag} ]]; then
	tag=iscomplex${is_complex}_${uuid}
fi
expdir=exp/train_dns_${tag}
mkdir -p $expdir && echo $uuid >> $expdir/run_uuid.txt
echo "Results from the following experiment will be stored in $expdir"


if [[ $stage -le  3 ]]; then
  echo "Stage 3 : Train"
  mkdir -p logs
  CUDA_VISIBLE_DEVICES=$id $python_path train.py \
		--json_dir $dumpdir \
		--is_complex $is_complex \
		--exp_dir ${expdir}/ | tee logs/train_${tag}.log
	cp logs/train_${tag}.log $expdir/train.log
fi

if [[ $stage -le  4 ]]; then
  echo "Stage 4 : Evaluate"
  $python_path eval_on_synthetic.py \
		--test_dir $clone_dir/DNS-Challenge/datasets/test_set/synthetic \
		--use_gpu $eval_use_gpu \
		--exp_dir $expdir | tee logs/eval_${tag}.log
	cp logs/eval_${tag}.log $expdir/eval.log
fi

if [[ $stage -le  5 ]]; then
  echo "Stage 5 : Separate"
  $python_path denoise.py \
		--denoise_path $clone_dir/DNS-Challenge/datasets/test_set/real_recordings/ \
		--use_gpu $eval_use_gpu \
		--exp_dir $expdir
fi
