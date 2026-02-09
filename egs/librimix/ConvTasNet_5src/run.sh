#!/bin/bash

# Exit on error
set -e
set -o pipefail

# Path to the python you'll use for the experiment.
python_path=python

# Example usage
# ./run.sh --stage 1 --tag my_tag --id 0,1

# General
stage=1  # Controls from which stage to start
tag=""  # Controls the directory name associated to the experiment
id=$CUDA_VISIBLE_DEVICES
out_dir=librimix

# Data config
task=sep_clean

# Training overrides (optional, defaults come from local/conf.yml)
epochs=200
batch_size=4

eval_use_gpu=1

. utils/parse_options.sh

# Stage 1: Prepare merged CSV data
if [[ $stage -le 1 ]]; then
  echo "Stage 1: Preparing merged variable-source CSV files"
  $python_path local/prepare_variable_data.py --task $task
fi

# Generate a random ID for the run if no tag is specified
uuid=$($python_path -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
if [[ -z ${tag} ]]; then
  tag=${uuid}
fi

expdir=exp/train_convtasnet_5src_${tag}
mkdir -p $expdir && echo $uuid >> $expdir/run_uuid.txt
echo "Results from the following experiment will be stored in $expdir"

# Stage 2: Training
if [[ $stage -le 2 ]]; then
  echo "Stage 2: Training"
  mkdir -p logs
  CUDA_VISIBLE_DEVICES=$id $python_path train.py \
    --exp_dir $expdir \
    --epochs $epochs \
    --batch_size $batch_size \
    --task $task | tee logs/train_${tag}.log
  cp logs/train_${tag}.log $expdir/train.log

  mkdir -p $expdir/publish_dir
  echo "librimix/ConvTasNet_5src" > $expdir/publish_dir/recipe_name.txt
fi

# Stage 3: Evaluation
if [[ $stage -le 3 ]]; then
  echo "Stage 3: Evaluation"
  $python_path eval.py \
    --exp_dir $expdir \
    --test_dir data/wav8k/min/test \
    --out_dir $out_dir \
    --use_gpu $eval_use_gpu \
    --task $task | tee logs/eval_${tag}.log
  cp logs/eval_${tag}.log $expdir/eval.log
fi

# Stage 4: ONNX export
if [[ $stage -le 4 ]]; then
  echo "Stage 4: ONNX export"
  $python_path export_onnx.py \
    --exp_dir $expdir
fi
