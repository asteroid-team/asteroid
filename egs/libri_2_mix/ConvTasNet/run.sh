#!/bin/bash
set -e  # Exit on error

# If you haven't downloaded and extracted the LibriSpeech dataset start from stage 0
# and specify storage_dir (only)

# If you have downloaded and extracted the LibriSpeech dataset start from stage 1
# and specify both librispeech_wav_dir and storage_dir.

# If you have generated LibriMix, start from stage 3 and specify librimix_wav_dir

# Main storage directory. You'll need disk space to store LibriSpeech and LibriMix
storage_dir=/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/mpariente/DATA/jcos_libri
# Directory where LibriSpeech is stored.

librispeech_dir=
#librispeech_dir=/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/mpariente/DATA/jcos_libri/LibriSpeech
#/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/mpariente/DATA/jcos_libri/LibriSpeech

# Directory where LibriMix is stored.
librimix_wav_dir=

# After running the recipe a first time, you can run it from stage 3 directly to train new models.

# Path to the python you'll use for the experiment. Defaults to the current python
# You can run ./utils/prepare_python_env.sh to create a suitable python environment, paste the output here.
#python_path=${storage_dir}/asteroid_conda/miniconda3/bin/python
python_path=python

# Example usage
# ./run.sh --stage 3 --tag my_tag --task sep_noisy --id 0,1

stage=1  # Controls from which stage to start


. utils/parse_options.sh


# General
#tag=""  # Controls the directory name associated to the experiment
## You can ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES)
#id=0,1,2,3
#
## Data
#data_dir=data  # Local data directory (No disk space needed)
#task=sep_clean  # Specify the task here (sep_clean, sep_noisy, enh_single, enh_both)
#sample_rate=16000
#
#
## Training
#batch_size=8
#num_workers=8
##optimizer=adam
#lr=0.001
#epochs=200
#
## Architecture
#n_blocks=6
#n_repeats=2
#mask_nonlinear=relu
#
## Evaluation
#eval_use_gpu=1
exp_dir=exp/tanh
test_dir=/home/jcosentino/libri/libri2mix/8K/min/metadata/mixture_test-clean.csv

if [[ $stage -le  0 ]]; then
	echo "Stage 0: Downloading LibriSpeech"
  . local/prepare_data.sh --storage_dir $storage_dir
	librispeech_dir=$storage_dir/LibriSpeech
fi

if [[ -z ${librispeech_dir} ]]; then
	librispeech_dir=$storage_dir/LibriSpeech
fi

if [[ $stage -le  1 ]]; then
	$python_path -m pip install pyloudnorm
	echo "Stage 1: Generating metadata "
	$python_path local/create_librispeech_metadata.py --librispeech_dir $librispeech_dir
  $python_path local/create_librimix_metadata.py --librispeech_dir $librispeech_dir --n_src 2
fi

metadata_dir=$librispeech_dir/metadata/librimix

if [[ -z ${librimix_wav_dir} ]]; then
	librimix_wav_dir=$librispeech_dir
fi

if [[ $stage -le  2 ]]; then
	echo "Stage 2: Generating Librimix dataset"
  $python_path local/create_librimix_from_metadata.py \
  --librispeech_dir $librispeech_dir \
  --metadata_dir $metadata_dir \
  --librimix_outdir $librimix_wav_dir \
  --n_src 2 \
  --freqs 8k
fi

## Generate a random ID for the run if no tag is specified
#uuid=$($python_path -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
#if [[ -z ${tag} ]]; then
#	tag=${task}_${sr_string}k${mode}_${uuid}
#fi
#expdir=exp/train_convtasnet_${tag}
#mkdir -p $expdir && echo $uuid >> $expdir/run_uuid.txt
#echo "Results from the following experiment will be stored in $expdir"


if [[ $stage -le 3 ]]; then
  echo "Stage 4: Training"
  mkdir -p logs
#  CUDA_VISIBLE_DEVICES=$id $python_path train.py --exp_dir exp/8K_mss \
#  --exp_dir ${expdir}/ | tee logs/train_${tag}.log
  $python_path train.py --exp_dir $exp_dir
fi


if [[ $stage -le 4 ]]; then
	echo "Stage 5 : Evaluation"
#	CUDA_VISIBLE_DEVICES=$id $python_path eval.py
  $python_path eval.py --exp_dir $exp_dir --test_dir $test_dir
fi
