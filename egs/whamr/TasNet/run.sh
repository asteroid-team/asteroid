#!/bin/bash

set -e  # Exit on error
# Main storage directory. You'll need disk space to dump the WHAM mixtures and the wsj0 wav
# files if you start from sphere files.
storage_dir=/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/mpariente/DATA/

# If you start from the sphere files, specify the path to the directory and start from stage 0
sphere_dir=  # Directory containing sphere files
# If you already have wsj0 wav files, specify the path to the directory here and start from stage 1
wsj0_wav_dir=${storage_dir}/wsj0_wav/
# If you already have the WHAM mixtures, specify the path to the directory here and start from stage 2
whamr_wav_dir=${storage_dir}/whamr_wav/
# After running the recipe a first time, you can run it from stage 3 directly to train new models.

# Path to the python you'll use for the experiment. Defaults to the current python
# You can run ./utils/prepare_python_env.sh to create a suitable python environment, paste the output here.
#python_path=${storage_dir}/asteroid_conda/miniconda3/bin/python
python_path=python

# Example usage
# ./run.sh --stage 3 --tag my_tag --task sep_noisy --id 0,1

# General
stage=1  # Controls from which stage to start
tag=""  # Controls the directory name associated to the experiment
# You can ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES)
id=

# Data
data_dir=data  # Local data directory (No disk space needed)
task=sep_clean  # Specify the task here (sep_clean, sep_noisy, enh_single, enh_both)
sample_rate=8000
mode=min
nondefault_src=  # If you want to train a network with 3 output streams for example.

# Training
batch_size=8
num_workers=8
#optimizer=adam
lr=0.001
epochs=200

# Architecture
n_blocks=8
n_repeats=3
mask_nonlinear=relu

# Evaluation
eval_use_gpu=1


. utils/parse_options.sh


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


if [[ $stage -le  0 ]]; then
  echo "Stage 0: Converting sphere files to wav files"
  . local/convert_sphere2wav.sh --sphere_dir $sphere_dir --wav_dir $wsj0_wav_dir
fi

if [[ $stage -le  1 ]]; then
	echo "Stage 1: Generating 8k and 16k WHAMR dataset"
  . local/prepare_data.sh --wav_dir $wsj0_wav_dir --out_dir $whamr_wav_dir --python_path $python_path
fi


if [[ $stage -le  2 ]]; then
	# Make json directories with min/max modes and sampling rates
	echo "Stage 2: Generating json files including wav path and duration"
	for sr_string in 8 16; do
		for mode in min max; do
			tmp_dumpdir=data/wav${sr_string}k/$mode
			echo "Generating json files in $tmp_dumpdir"
			[[ ! -d $tmp_dumpdir ]] && mkdir -p $tmp_dumpdir
			local_wham_dir=$whamr_wav_dir/wav${sr_string}k/$mode/
      $python_path local/preprocess_whamr.py --in_dir $local_wham_dir --out_dir $tmp_dumpdir
    done
  done
fi

sr_string=$(($sample_rate/1000))
suffix=wav${sr_string}k/$mode
dumpdir=data/$suffix  # directory to put generated json file

train_dir=$dumpdir/tr
valid_dir=$dumpdir/cv
test_dir=$dumpdir/tt

# Generate a random ID for the run if no tag is specified
uuid=$($python_path -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
if [[ -z ${tag} ]]; then
	tag=${task}_${sr_string}k${mode}_${uuid}
fi
expdir=exp/train_convtasnet_${tag}
mkdir -p $expdir && echo $uuid >> $expdir/run_uuid.txt
echo "Results from the following experiment will be stored in $expdir"

#
#if [[ $stage -le 3 ]]; then
#  echo "Stage 3: Training"
#  mkdir -p logs
#  CUDA_VISIBLE_DEVICES=$id $python_path train.py \
#  --train_dir $train_dir \
#  --valid_dir $valid_dir \
#  --task $task \
#  --sample_rate $sample_rate \
#  --lr $lr \
#  --epochs $epochs \
#  --batch_size $batch_size \
#  --num_workers $num_workers \
#  --mask_act $mask_nonlinear \
#  --n_blocks $n_blocks \
#  --n_repeats $n_repeats \
#  --exp_dir ${expdir}/ | tee logs/train_${tag}.log
#fi
#
#
#if [[ $stage -le 4 ]]; then
#	echo "Stage 4 : Evaluation"
#	CUDA_VISIBLE_DEVICES=$id $python_path eval.py \
#	--task $task \
#	--test_dir $test_dir \
#	--use_gpu $eval_use_gpu \
#	--exp_dir ${expdir}
#fi
