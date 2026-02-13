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
tag=$TAG  # Controls the directory name associated to the experiment
id=$CUDA_VISIBLE_DEVICES
out_dir=librimix
debug_profile=

# Data config
task=sep_clean
dataset_type=online_mix
min_speakers=2
max_speakers=5
speaker_count_weights=0.10,0.15,0.35,0.40

# Training overrides (optional, defaults come from local/conf.yml)
epochs=200
batch_size=4
num_examples=
val_num_examples=

# Loss config overrides
loss_mode=active_only_pit
silence_weight=0.2
silence_margin_db=-45.0
silence_metric=rms_db
active_threshold_mode=rms
debug_log_components=false
early_stop_patience=30
early_stop_min_delta=0.0
plot_metrics=true
plot_metric_prefixes=val_,test_
use_wandb=true
wandb_project=librimix-convtasnet-5src
wandb_entity=
wandb_group=
wandb_run_name=
wandb_tags=
wandb_job_type=train
wandb_watch_model=false

eval_use_gpu=1

. utils/parse_options.sh

if [[ $debug_profile == "two_spk_in_5ch" ]]; then
  dataset_type=online_mix
  min_speakers=2
  max_speakers=2
  speaker_count_weights=1.0
  epochs=10
  batch_size=4
  num_examples=2000
  val_num_examples=400
fi

# Stage 1: Prepare merged CSV data
if [[ $stage -le 1 ]]; then
  if [[ $dataset_type == "variable_librimix" ]]; then
    echo "Stage 1: Preparing merged variable-source CSV files"
    $python_path local/prepare_variable_data.py --task $task
  else
    echo "Stage 1: Skipping CSV preparation for online_mix"
  fi
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
    --dataset_type $dataset_type \
    --task $task \
    --min_speakers $min_speakers \
    --max_speakers $max_speakers \
    --speaker_count_weights "$speaker_count_weights" \
    --mode $loss_mode \
    --silence_weight $silence_weight \
    --silence_margin_db $silence_margin_db \
    --silence_metric $silence_metric \
    --active_threshold_mode $active_threshold_mode \
    --debug_log_components $debug_log_components \
    --early_stop_patience $early_stop_patience \
    --early_stop_min_delta $early_stop_min_delta \
    --plot_metrics $plot_metrics \
    --plot_metric_prefixes "$plot_metric_prefixes" \
    --use_wandb $use_wandb \
    --wandb_project "$wandb_project" \
    --wandb_job_type "$wandb_job_type" \
    --wandb_watch_model $wandb_watch_model \
    ${wandb_entity:+--wandb_entity "$wandb_entity"} \
    ${wandb_group:+--wandb_group "$wandb_group"} \
    ${wandb_run_name:+--wandb_run_name "$wandb_run_name"} \
    ${wandb_tags:+--wandb_tags "$wandb_tags"} \
    ${num_examples:+--num_examples $num_examples} \
    ${val_num_examples:+--val_num_examples $val_num_examples} | tee logs/train_${tag}.log
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
