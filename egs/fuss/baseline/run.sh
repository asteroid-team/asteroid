#!/bin/bash

# Main storage directory (80Go at least without data augmentation)
storage_dir=/mnt/beegfs/pul51/zaf67/DATA/FUSS_DEV/

# If you want to create the dataset from scratch or perform data augmentation
# set the following variables and start from stage 0.
# INSTALL sox and ffmpeg in this case: conda install -c conda-forge sox ffmpeg
augment_data=false
num_train=20000  # Discarded if augment_data=false
num_val=2000     # Discarded if augment_data=false
augment_random_seed=2020  # Discarded if augment_data=false

# If you already have FUSS wav files, specify the path to the root of
# the directory here and start from stage 2
fuss_dir=

# Path to the python you'll use for the experiment. Defaults to the current python
# You can run ./utils/prepare_python_env.sh to create a python environment, paste the output here.
python_path=python

# Example usage
# ./run.sh --stage 0 --tag my_tag --task sep_reverb --id 0,1

# General
stage=0  # Controls from which stage to start
tag=""  # Controls the directory name associated to the experiment
# You can ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES)
id=

# Data
task=sep_reverb  # One of sep_dry, sep_reverb

# Training
batch_size=2
lr=0.001
epochs=200
# Architecture
improved=y  # TDCN++ or TDCN: true or false
n_blocks=6
n_repeats=2
# Evaluation
eval_use_gpu=1

# End configuration section
. utils/parse_options.sh

# you might not want to do this for interactive shells.
set -e

# Generate a random ID for the run if no tag is specified
uuid=$($python_path -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
if [[ -z ${tag} ]]; then
	tag=${task}_${uuid}
fi
expdir=exp/train_tdcnpp_${tag}


# All the data generation scripts rely on ROOT_DIR
export ROOT_DIR=$storage_dir

# Download necessary data
if [[ $stage -le 0 ]]; then
    echo "Stage 0: Download mixtures"
    git clone https://github.com/google-research/sound-separation local/sound-separation
    cp -r ./local/sound-separation/datasets/fuss/ local/
    # setup.sh defines everything relative to ROOT_DIR
    . ./local/fuss/setup.sh
    # Download fixed train/val sets
    bash local/fuss/get_dev_data.sh
	if [[ $augment_data == "true" ]]; then
    	echo "Download single sources for data augmentation"
			# Those variable are defined in setup.sh above
			bash local/fuss/get_raw_data.sh ${DOWNLOAD_DIR} ${RAW_DATA_DIR} ${FSD_DATA_URL} ${RIR_DATA_URL}
	fi
fi


# Data augmentation stage
if [[ $stage -le 1 ]]; then
	if [[ $augment_data == "true" ]]; then
		echo "Stage 1: Run data augmentation"
		if ! [[ -x "$(command -v ffmpeg)" ]]; then
			echo 'Error: ffmpeg is not installed and is necessary to run data augmentation, please install it using :'
			echo 'conda install -c conda-forge ffmpeg / sudo apt-get install ffmpeg / brew install ffmpeg'
			exit 1
		fi
		if ! [[ -x "$(command -v sox)" ]]; then
			echo 'Error: sox is not installed and is necessary to run data augmentation, please install it using :'
			echo 'conda install -c conda-forge sox / sudo apt-get install sox / brew install sox'
			exit 1
		fi
		# Replace variables in setup.sh by the one provided here.
		sed -i 's/NUM_TRAIN=.*/NUM_TRAIN='"$num_train"'/g' local/fuss/setup.sh
		sed -i 's/NUM_VAL=.*/NUM_VAL='"$num_val"'/g' local/fuss/setup.sh
		sed -i 's/RANDOM_SEED=.*/RANDOM_SEED='"$augment_random_seed"'/g' local/fuss/setup.sh
		# Replace pip3 in install_deps
		sed -i 's+pip3+'"$python_path"' -m pip+g'
		# Run original data augmentation script
    bash ./local/fuss/run_data_augmentation.sh
	else
		echo "Stage 1: Passing because augment_data!=true"
  fi
fi


if [[ $stage -le 2 ]]; then
    echo "Stage 2: Generating csv files including absolute wav path"
    # Process augmented folder only if augmentation was done
    aug_folder=
    if [[ $augment_data == "true" ]]; then
    	aug_folder=fuss_augment_${augment_random_seed}
		fi
		# Copy sound-separation output locally. Replace rel by abs path.
		for fold in fuss_dev $aug_folder; do
			for subfold in ssdata ssdata_reverb; do
				mkdir -p data/${fold}/${subfold}/
				common=${fold}/${subfold}
				# Train and validation are copied for each seed of augmentation
				for split in train validation; do
					cp ${storage_dir}/${common}/${split}_example_list.txt data/${common}
					# Absolute path instead of relative
					sed -i 's+'"$split"'+'"${storage_dir}${common}/${split}"'+g' data/${common}/${split}_example_list.txt
				# The eval set is always the same, so we copy it as eval_example_list.txt in
				# the augmented directories, but it is the same.
				cp ${storage_dir}/fuss_dev/${subfold}/eval_example_list.txt data/${common}
#				sed -i 's+'"$split"'+'"${storage_dir}${common}${split}"'+g' data/${common}/${split}_example_list.txt
				done
			done
		done
fi

if [[ $stage -le 3 ]]; then
	echo "Stage 3: Training"
	mkdir -p logs
	# Find the data folder. Data augmentation or not
	if [[ $augment_data == "true" ]]; then
		exp_data=data/fuss_augment_${augment_random_seed}
	else
		exp_data=data/fuss_dev
	fi

	# Dry or reverberated separation
	exp_data=${exp_data}/ssdata
	if [[ $task == "sep_reverb" ]]; then
		exp_data=${exp_data}_reverb
	fi

	mkdir -p $expdir && echo $uuid >> $expdir/run_uuid.txt
	echo "Results from the following experiment will be stored in $expdir"

	CUDA_VISIBLE_DEVICES=$id $python_path train.py \
		--train_list $exp_data/train_example_list.txt \
		--valid_list $exp_data/validation_example_list.txt \
		--lr $lr \
		--epochs $epochs \
		--batch_size $batch_size \
		--improved $improved \
		--n_blocks $n_blocks \
		--n_repeats $n_repeats \
		--exp_dir ${expdir}/ | tee logs/train_${tag}.log
fi


if [[ $stage -le 4 ]]; then
	echo "Stage 4 : Evaluation"
	CUDA_VISIBLE_DEVICES=$id $python_path eval.py \
	--test_file $exp_data/eval_example_list.txt \
	--use_gpu $eval_use_gpu \
	--exp_dir ${expdir} | tee logs/eval_${tag}.log
fi
