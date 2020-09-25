#!/bin/bash
set -e  # Exit on error

#if starting from stage 0
# Destination to save json files with list of track locations for instrument sets
json_dir=/homes/ss404/projects/temp
# Location for tracklist for all data dirs
tracklist=  # Directory containing tracklists for V1, V2, Bach10 and others

# Location for MedleyDB V1
V1_dir="/import/research_c4dm/ss404/V1" # Directory containing MedleyDB V1 audio files

# Location for MedleyDB V2
V2_dir="/import/research_c4dm/ss404/V2" # Directory containing MedleyDB V2 audio files

# Location for Bach10
Bach10_dir=  # Directory containing MedleyDB format Bach10 audio files

# Location for additional MedleyDB format multitracks
extra_dir= # Directory containing additional MedleyDB format audio files

# Location for MedleyDB format metadata files for all multitracks
metadata_dir=/homes/ss404/projects/temp/mdb/ # Directory containing MedleyDB github repository with metadata for all files

# Location for evaluation multitrack sourceFolders
wav_dir=/homes/ss404/projects/Conv-TasNet/data/full/tt # Directory containing MedleyDB github repository with metadata for all files



# After running the recipe a first time, you can run it from stage 3 directly to train new models.

# Path to the python you'll use for the experiment. Defaults to the current python
# You can run ./utils/prepare_python_env.sh to create a suitable python environment, paste the output here.
python_path=python

# Example usage
# ./run.sh --stage 3 --tag my_tag --loss_alpha 0.1 --id 0,1

# General
stage=3  # Controls from which stage to start
tag="test_8k"  # Controls the directory name associated to the experiment
# You can ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES)
id=$CUDA_VISIBLE_DEVICES

# Data
#data_dir=data  # Local data directory (No disk space needed)
sample_rate=8000
n_inst=1  # 2 or 3
n_poly=2
segment=5.0
# Training
batch_size=10
num_workers=10
optimizer=rmsprop
lr=0.0003
weight_decay=0.0
epochs=1
loss_alpha=1.0  # DC loss weight : 1.0 => DC, <1.0 => Chimera
take_log=true  # Whether to input log mag spec to the NN

# Evaluation
eval_use_gpu=1

. utils/parse_options.sh

sr_string=$(($sample_rate/1000))
suffix=${n_inst}inst${n_poly}poly${sr_string}sr${segment}sec
dumpdir=data/$suffix  # directory to put generated json file

#json_dir=$dumpdir
is_raw=True

if [[ $stage -le  0 ]]; then
  echo "Stage 0 : Downloading MedleyDB repo for tracklist"
  mkdir -p $metadata_dir
  git clone https://github.com/marl/medleydb.git  $metadata_dir
  fi

if [[ $stage -le  1 ]]; then
  echo "Stage 1: Download MedleyDB dataset, update dataset path, add custom tracklist if required"
fi

if [[ $stage -le  2 ]]; then
	# Make json files with wav paths for instrument set
	echo "Stage 2: Generating json files including wav path and activity info"
  $python_path local/preprocess_medleyDB.py --metadata_path $metadata_dir --json_dir $json_dir --v1_path "$V1_dir" --v2_path "$V2_dir"

fi

# Generate a random ID for the run if no tag is specified
uuid=$($python_path -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
if [[ -z ${tag} ]]; then
	tag=${n_src}sep_${sr_string}k${mode}_${uuid}
fi
expdir=/homes/ss404/projects/asteroid/egs/MedleyDB/ConvTasNet/exp/train_convtasnet_demucs_orig_44_lr3_15stop
if [[ $stage -le 3 ]]; then
  echo "Stage 3: Training"
  expdir=exp/train_convtasnet_${tag}
  mkdir -p $expdir && echo $uuid >> $expdir/run_uuid.txt
  echo "Results from the following experiment will be stored in $expdir"

  mkdir -p logs
  CUDA_VISIBLE_DEVICES=$id $python_path train.py \
		--json_dir $json_dir \
		--n_inst $n_inst \
		--sample_rate $sample_rate \
		--optimizer $optimizer \
		--lr $lr \
		--weight_decay $weight_decay \
		--epochs $epochs \
		--batch_size $batch_size \
		--num_workers $num_workers \
		--exp_dir ${expdir}/ | tee logs/train_${tag}.log
	cp logs/train_${tag}.log $expdir/train.log
fi

if [[ $stage -le 4 ]]; then
	echo "Stage 4 : Evaluation"
	CUDA_VISIBLE_DEVICES=$id $python_path eval.py \
	  --n_src $n_poly \
		--wav_dir $wav_dir \
		--use_gpu $eval_use_gpu \
		--batch_size $batch_size \
		--exp_dir ${expdir} | tee logs/eval_${tag}.log
	cp logs/eval_${tag}.log $expdir/eval.log
fi
