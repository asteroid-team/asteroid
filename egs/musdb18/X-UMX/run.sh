#!/bin/bash

# Exit on error
set -e
set -o pipefail

# Main storage directory.
# If you start from downloading MUSDB18, you'll need disk space to dump the MUSDB18 and its wav.
musdb18_dir=data

# After running the recipe a first time, you can run it from stage 1 directly to train new models.

# Path to the python you'll use for the experiment. Defaults to the current python
# You can run ./utils/prepare_python_env.sh to create a suitable python environment, paste the output here.
python_path=python

# Example usage
# ./run.sh --stage 1 --tag my_tag --mix_coef 10.0

# General
stage=0  # Controls from which stage to start
tag=  # Controls the directory name associated to the experiment
# You can ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES)
id=$CUDA_VISIBLE_DEVICES

# Data
sample_rate=44100
seed=42
seq_dur=6.0
samples_per_track=64

# Training
batch_size=14
num_workers=4
optimizer=adam
lr=0.001
weight_decay=0.00001
patience=1000
lr_decay_patience=80
lr_decay_gamma=0.3
epochs=1000
mix_coef=10.0
val_dur=80.0

# Evaluation
eval_use_gpu=-1

. utils/parse_options.sh

if [[ $stage -le  0 ]]; then
  echo "Stage 0: Downloading MUSDB18 into $musdb18_dir"
  pip install -r requirements.txt
  wget -c --tries=0 --read-timeout=20 https://zenodo.org/record/1117372/files/musdb18.zip -P $musdb18_dir
  mkdir -p $musdb18_dir/logs
  unzip $musdb18_dir/musdb18.zip -d $musdb18_dir >> $musdb18_dir/logs/unzip_musdb18.log
  musdbconvert $musdb18_dir $musdb18_dir
fi

# Generate a random ID for the run if no tag is specified
uuid=$($python_path -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
if [[ -z ${tag} ]]; then
	tag=${uuid}
fi
expdir=exp/train_xumx_${tag}
mkdir -p $expdir && echo $uuid >> $expdir/run_uuid.txt
echo "Results from the following experiment will be stored in $expdir"

if [[ $stage -le 1 ]]; then
  echo "Stage 1: Training"
  mkdir -p logs
  CUDA_VISIBLE_DEVICES=$id $python_path train.py \
  --train_dir $musdb18_dir \
  --sample_rate $sample_rate \
  --seed $seed \
  --seq_dur $seq_dur \
  --samples_per_track $samples_per_track \
  --batch_size $batch_size \
  --num_workers $num_workers \
  --optimizer $optimizer \
  --lr $lr \
  --weight_decay $weight_decay \
  --patience $patience \
  --lr_decay_patience $lr_decay_patience \
  --lr_decay_gamma $lr_decay_gamma \
  --epochs $epochs \
  --mix_coef $mix_coef \
  --val_dur $val_dur \
  --output ${expdir} | tee logs/train_${tag}.log
	cp logs/train_${tag}.log $expdir/train.log
fi

if [[ $stage -le 2 ]]; then
	echo "Stage 2: Evaluation"
  if [[ $eval_use_gpu -lt 0 ]]; then
    $python_path eval.py \
    --no-cuda \
    --root $musdb18_dir \
    --outdir ${expdir} | tee logs/eval_${tag}.log
  else
    CUDA_VISIBLE_DEVICES=$id $python_path eval.py \
    --root $musdb18_dir \
    --outdir ${expdir} | tee logs/eval_${tag}.log
  fi
  cp logs/eval_${tag}.log $expdir/eval.log
fi
