#!/bin/bash
set -e  # Exit on error

# If you haven't generated LibriMix start from stage 0
# Main storage directory. You'll need disk space to store LibriSpeech, WHAM noises
# and LibriMix. This is about 500 Gb
storage_dir=

# After running the recipe a first time, you can run it from stage 3 directly to train new models.

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
out_dir=librimix # Controls the directory name associated to the evaluation results inside the experiment directory

# Network config
n_blocks=8
n_repeats=3
mask_act=relu
# Training config
epochs=200
batch_size=24
num_workers=4
half_lr=yes
early_stop=yes
# Optim config
optimizer=adam
lr=0.001
weight_decay=0.
# Data config
train_dir=data/wav8k/min/train-360
valid_dir=data/wav8k/min/dev
test_dir=data/wav8k/min/test
sample_rate=8000
n_src=2
segment=3
task=sep_clean

. utils/parse_options.sh


if [[ $stage -le  0 ]]; then
	echo "Stage 0: Generating Librimix dataset"
  . local/prepare_data.sh --storage_dir $storage_dir --n_src $n_src
fi

# Generate a random ID for the run if no tag is specified
uuid=$($python_path -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
if [[ -z ${tag} ]]; then
	tag=${uuid}
fi

expdir=exp/train_convtasnet_${tag}
mkdir -p $expdir && echo $uuid >> $expdir/run_uuid.txt
echo "Results from the following experiment will be stored in $expdir"


if [[ $stage -le 1 ]]; then
  echo "Stage 1: Training"
  mkdir -p logs
  CUDA_VISIBLE_DEVICES=$id $python_path train.py --exp_dir $expdir \
		--n_blocks $n_blocks \
		--n_repeats $n_repeats \
		--mask_act $mask_act \
		--epochs $epochs \
		--batch_size $batch_size \
		--num_workers $num_workers \
		--half_lr $half_lr \
		--early_stop $early_stop \
		--optimizer $optimizer \
		--lr $lr \
		--weight_decay $weight_decay \
		--train_dir $train_dir \
		--valid_dir $valid_dir \
		--sample_rate $sample_rate \
		--n_src $n_src \
		--segment $segment | tee logs/train_${tag}.log
	cp logs/train_${tag}.log $expdir/train.log
fi

if [[ $stage -le 2 ]]; then
	echo "Stage 2 : Evaluation"
  $python_path eval.py --exp_dir $expdir --test_dir $test_dir \
  	--out_dir $out_dir \
  	--task $task | tee logs/eval_${tag}.log
	cp logs/eval_${tag}.log $expdir/eval.log
fi
