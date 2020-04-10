#!/bin/bash
set -e  # Exit on error

# If you haven't downloaded and extracted the LibriSpeech dataset start from stage 0
# and specify storage_dir (only)
# Main storage directory. You'll need disk space to store LibriSpeech and LibriMix
storage_dir=

# If you have downloaded and extracted the LibriSpeech dataset (stage 0) start from stage 1
# and specify both librispeech_dir and storage_dir.
# Directory where LibriSpeech is stored.
librispeech_dir=

# If you have generated LibriMix metadata (stage 1) but you haven't generated the dataset start from stage 2
# and specify storage_dir, librispeech_dir and metadata_dir
metadata_dir=

# After running the recipe a first time, you can run it from stage 3 directly to train new models.

# Path to the python you'll use for the experiment. Defaults to the current python
# You can run ./utils/prepare_python_env.sh to create a suitable python environment, paste the output here.
#python_path=${storage_dir}/asteroid_conda/miniconda3/bin/python
python_path=python

# All the parameters
# General
tag=""  # Controls the directory name associated to the experiment

# Network config
n_blocks=6
n_repeats=2
mask_act=relu
# Training config
epochs=1
batch_size=4
num_workers=2
half_lr=yes
early_stop=yes
# Optim config
optimizer=adam
lr=0.001
weight_decay=0.
# Data config
train_dir=data/wav8k/min/dev
valid_dir=data/wav8k/min/dev
test_dir=data/8k/min/test
sample_rate=8000
n_src=2
segment=4


stage=3  # Controls from which stage to start

. utils/parse_options.sh

# Example usage
# ./run.sh --stage 3 --tag my_tag --task sep_noisy --id 0,1

echo $stage
if [[ $stage -le  0 ]]; then
	echo "Stage 0: Downloading LibriSpeech"
  . local/prepare_data.sh --storage_dir $storage_dir
	librispeech_dir=$storage_dir/LibriSpeech
fi

if [[ $stage -le  1 ]]; then
	$python_path -m pip install pyloudnorm
	echo "Stage 1: Generating metadata "
	$python_path local/create_librispeech_metadata.py --librispeech_dir $librispeech_dir
  $python_path local/create_librimix_metadata.py --librispeech_dir $librispeech_dir  --n_src 2
  metadata_dir=$librispeech_dir/../LibriMix/metadata
fi

if [[ $stage -le  2 ]]; then
	echo "Stage 2: Generating Librimix dataset"
  $python_path local/create_librimix_from_metadata.py \
  --librispeech_dir $librispeech_dir \
  --metadata_dir $metadata_dir \
  --n_src 2 \
  --freqs 8k \
  --modes min
fi

# Generate a random ID for the run if no tag is specified
uuid=$($python_path -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
if [[ -z ${tag} ]]; then
	tag=${uuid}
fi
exp_dir=exp/train_convtasnet_${tag}
mkdir -p $exp_dir && echo $uuid >> $exp_dir/run_uuid.txt
echo "Results from the following experiment will be stored in $exp_dir"


if [[ $stage -le 3 ]]; then
  echo "Stage 3: Training"
  mkdir -p logs
  $python_path train.py --exp_dir $exp_dir \
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
  --segment $segment
fi

if [[ $stage -le 4 ]]; then
	echo "Stage 4 : Evaluation"
  $python_path eval.py --exp_dir $exp_dir --test_dir $test_dir
fi
