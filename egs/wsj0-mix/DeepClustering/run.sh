#!/bin/bash

set -e  # Exit on error
# Main storage directory. You'll need disk space to dump the WHAM mixtures and the wsj0 wav
# files if you start from sphere files.
storage_dir=/media/sam/cb915f0e-e440-414c-bb74-df66b311d09d/
#storage_dir=/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/mpariente/DATA/wsj0_wav


# If you start from the sphere files, specify the path to the directory and start from stage 0
sphere_dir=  # Directory containing sphere files
# If you already have wsj0 wav files, specify the path to the directory here and start from stage 1
wsj0_wav_dir=${storage_dir}/wsj0_wav/
# If you already have the WHAM mixtures, specify the path to the directory here and start from stage 2
#wsj0mix_wav_dir=${storage_dir}/wsj0_mix/
#wsj0mix_wav_dir=/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/mpariente/DATA/wsj0_wav
wsj0mix_wav_dir=/mnt/beegfs/pul51/zaf67/DATA/wsj0-mix

# After running the recipe a first time, you can run it from stage 3 directly to train new models.

# Path to the python you'll use for the experiment. Defaults to the current python
# You can run ./utils/prepare_python_env.sh to create a suitable python environment, paste the output here.
#python_path=${storage_dir}/asteroid_conda/miniconda3/bin/python
python_path=python

# Example usage
# ./run.sh --stage 3 --tag my_tag --task sep_noisy --id 0,1

# General
stage=3  # Controls from which stage to start
tag=""  # Controls the directory name associated to the experiment
# You can ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES)
id=

# Data
#data_dir=data  # Local data directory (No disk space needed)
sample_rate=8000
mode=min
n_src=2  # 2 or 3

# Training
batch_size=64
num_workers=8
#optimizer=adam
lr=0.00001
epochs=200
loss_alpha=1.0
take_log=true

# Evaluation
eval_use_gpu=1


. utils/parse_options.sh


if [[ $stage -le  0 ]]; then
  echo "Stage 0: Converting sphere files to wav files"
  . local/convert_sphere2wav.sh --sphere_dir $sphere_dir --wav_dir $wsj0_wav_dir
fi


if [[ $stage -le  1 ]]; then
	echo "Stage 1 : You need to generate the wsj0-mix dataset using the official scripts."
	# Link + WHAM is ok for 2 source.
	exit
fi


if [[ $stage -le  2 ]]; then
	# Make json directories with min/max modes and sampling rates
	echo "Stage 2: Generating json files including wav path and duration"
	for sr_string in 8 16; do
		for mode_option in min max; do
			for tmp_nsrc in 2 3; do
				tmp_dumpdir=data/${tmp_nsrc}speakers/wav${sr_string}k/$mode_option
				echo "Generating json files in $tmp_dumpdir"
				[[ ! -d $tmp_dumpdir ]] && mkdir -p $tmp_dumpdir
				local_wsj_dir=$wsj0mix_wav_dir/${tmp_nsrc}speakers/wav${sr_string}k/$mode_option/
				$python_path local/preprocess_wsj0mix.py --in_dir $local_wsj_dir \
				 																			--n_src $tmp_nsrc \
				 																			--out_dir $tmp_dumpdir
			done
    done
  done
fi


sr_string=$(($sample_rate/1000))
suffix=${n_src}speakers/wav${sr_string}k/$mode
dumpdir=data/$suffix  # directory to put generated json file

train_dir=$dumpdir/tr
valid_dir=$dumpdir/cv
test_dir=$dumpdir/tt

# Generate a random ID for the run if no tag is specified
uuid=$($python_path -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
if [[ -z ${tag} ]]; then
	tag=${n_src}sep_${sr_string}k${mode}_${uuid}
fi
expdir=exp/train_chimera_${tag}
mkdir -p $expdir && echo $uuid >> $expdir/run_uuid.txt
echo "Results from the following experiment will be stored in $expdir"


if [[ $stage -le 3 ]]; then
  echo "Stage 3: Training"
  mkdir -p logs
  CUDA_VISIBLE_DEVICES=$id $python_path train.py \
		--train_dir $train_dir \
		--valid_dir $valid_dir \
		--n_src $n_src \
		--sample_rate $sample_rate \
		--lr $lr \
		--epochs $epochs \
		--batch_size $batch_size \
		--loss_alpha $loss_alpha \
		--take_log $take_log \
		--num_workers $num_workers \
		--exp_dir ${expdir}/ | tee logs/train_${tag}.log
fi

if [[ $stage -le 4 ]]; then
	echo "Stage 4 : Evaluation"
	CUDA_VISIBLE_DEVICES=$id $python_path eval.py \
	  --n_src $n_src \
		--test_dir $test_dir \
		--use_gpu $eval_use_gpu \
		--exp_dir ${expdir}
fi
