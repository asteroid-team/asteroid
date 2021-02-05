#!/bin/bash

# Exit on error
set -eo pipefail

# General
stage=0  # Controls from which stage to start
tag=1a_wham  # Controls the directory name associated to the experiment

# GPU setting.
# You can ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES)
#   For single gpu, "0" or "1" would work
#   For multi-gpu, something like "0,1"
id=0

# Data
task=sep_clean  # Specify the task here (sep_clean, sep_noisy, enh_single, enh_both)
sample_rate=8000
mode=min
nondefault_src=  # If you want to train a network with 3 output streams for example.

# Training
batch_size=4
num_workers=8
optimizer=adam
lr=0.001
epochs=200

# Architecture
n_blocks=8
n_repeats=3
mask_nonlinear=relu

# Evaluation
eval_use_gpu=1

# Main storage directory. You'll need disk space to dump the WHAM mixtures and the wsj0 wav
# files if you start from sphere files.
storage_dir=corpus
# If you start from the sphere files, specify the path to the directory and start from stage 0
sphere_dir=$storage_dir/sph_files
# If you already have wsj0 wav files, specify the path to the directory here and start from stage 1
wsj0_wav_dir=$storage_dir/wsj0_wavs
# If you already have the WHAM mixtures, specify the path to the directory here and start from stage 2
wham_wav_dir=$storage_dir/wham_wavs

. utils/parse_options.sh

# After running the recipe a first time, you can run it from stage 3 directly to train new models.
mkdir -p $sphere_dir $wsj0_wav_dir $wham_wav_dir

sr_string=$(($sample_rate/1000))
suffix=wav${sr_string}k/$mode
dumpdir=data/$suffix  # directory to put generated json file

train_dir=$dumpdir/tr
valid_dir=$dumpdir/cv
test_dir=$dumpdir/tt

# Generate a random ID for the run if no tag is specified
uuid=$(python3 -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
if [[ -z ${tag} ]]; then
	tag=${task}_${uuid}
fi
expdir=exp/train_convtasnet_${tag}
mkdir -p $expdir && echo $uuid >> $expdir/run_uuid.txt
echo "Results from the following experiment will be stored in $expdir"


if [ $stage -le 0 ]; then
    echo "Stage 0: Converting sphere files to wav files"
    local/convert_sphere2wav.sh --sphere_dir $sphere_dir --wav_dir $wsj0_wav_dir
fi

if [ $stage -le 1 ]; then
    echo "Stage 1: Generating 8k and 16k WHAM dataset"
    local/prepare_data.sh --wav-dir $wsj0_wav_dir --out-dir $wham_wav_dir
fi

if [ $stage -le 2 ]; then
	# Make json directories with min/max modes and sampling rates
	echo "Stage 2: Generating json files including wav path and duration"
	for sr_string in 8 16; do
		for mode_option in min max; do
			tmp_dumpdir=data/wav${sr_string}k/$mode_option
			echo "Generating json files in $tmp_dumpdir"
			[[ ! -d $tmp_dumpdir ]] && mkdir -p $tmp_dumpdir
			local_wham_dir=$wham_wav_dir/wav${sr_string}k/$mode_option/
            python3 local/preprocess_wham.py --in_dir $local_wham_dir --out_dir $tmp_dumpdir
        done
    done
fi

if [ $stage -le 3 ]; then
    echo "Stage 3: Training"
    mkdir -p logs
    CUDA_VISIBLE_DEVICES=$id python3 train.py \
        --train_dir $train_dir \
        --valid_dir $valid_dir \
        --task $task \
        --sample_rate $sample_rate \
        --lr $lr \
        --epochs $epochs \
        --batch_size $batch_size \
        --num_workers $num_workers \
        --mask_act $mask_nonlinear \
        --n_blocks $n_blocks \
        --n_repeats $n_repeats \
        --exp_dir ${expdir}/ | tee logs/train_${tag}.log
    cp logs/train_${tag}.log $expdir/train.log

    # Get ready to publish
    mkdir -p $expdir/publish_dir
    echo "wham/ConvTasNet" > $expdir/publish_dir/recipe_name.txt
fi

if [ $stage -le 4 ]; then
	echo "Stage 4 : Evaluation"
	CUDA_VISIBLE_DEVICES=$id python3 eval.py \
		--task $task \
		--test_dir $test_dir \
		--use_gpu $eval_use_gpu \
		--exp_dir ${expdir} | tee logs/eval_${tag}.log
	cp logs/eval_${tag}.log $expdir/eval.log
fi
