#!/bin/bash

set -e  # Exit on error
set -o pipefail

# Path to the python you'll use for the experiment. Defaults to the current python
# You can run ./utils/prepare_python_env.sh to create a suitable python environment, paste the output here.
python_path=python

# Example usage
# ./run.sh --stage 3 --tag my_tag

# General
stage=1  # Controls from which stage to start
tag=""  # Controls the directory name associated to the experiment
# You can ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES)
id=$CUDA_VISIBLE_DEVICES
out_dir=dampvsep # Controls the directory name associated to the evaluation results inside the experiment directory

# Network config
n_blocks=10
n_repeats=4
mask_act=relu

# Training config
epochs=200
batch_size=7
num_workers=10
half_lr=yes
early_stop=yes
loss_alpha=0.5

# Optim config
optimizer=adam
lr=0.0005
weight_decay=0.

# Data config
task=separation  #'enh_vocal', 'separation'
root_path=        # full path to DAMP-VSEP
mixture=remix     # 'original' includes non-linear effects,
                  # 'remix' add both sources together in runtime
train_set=english   # 'english' uses the english subset ~77 hrs
                    # 'singles' uses all singles performances subset ~174 hrs
segment=3.0
ex_per_track=5   # Number of examples obtained per track
sample_rate=16000

. utils/parse_options.sh

if [[ $stage -le  0 ]]; then
	echo "Stage 0: Generating preprocessed DAMPVSEP dataset"
  . local/prepare_data.sh  --dampvsep_root $root_path
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
  CUDA_VISIBLE_DEVICES=$id $python_path train.py \
        --exp_dir $expdir \
        --n_blocks $n_blocks \
        --n_repeats $n_repeats \
        --mask_act $mask_act \
        --epochs $epochs \
        --batch_size $batch_size \
        --num_workers $num_workers \
        --half_lr $half_lr \
        --early_stop $early_stop \
        --loss_alpha $loss_alpha \
        --optimizer $optimizer \
        --lr $lr \
        --weight_decay $weight_decay \
        --task $task \
        --root_path $root_path \
        --mixture $mixture \
        --sample_rate $sample_rate \
        --train_set $train_set \
        --segment $segment \
        --ex_per_track $ex_per_track | tee logs/train_${tag}.log
	cp logs/train_${tag}.log $expdir/train.log


	# Get ready to publish
	mkdir -p $expdir/publish_dir
	echo "damp-vsep/ConvTasNet" > $expdir/publish_dir/recipe_name.txt
fi


if [[ $stage -le 2 ]]; then
   echo "Stage 2 : Evaluation"
  $python_path -u eval.py --exp_dir $expdir --use_gpu 1 \
  	--out_dir $out_dir --n_save_ex 5 | tee logs/eval_${tag}.log
	cp logs/eval_${tag}.log $expdir/eval.log
fi
