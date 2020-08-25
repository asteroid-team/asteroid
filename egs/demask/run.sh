#!/bin/bash

# Exit on error
set -e
set -o pipefail

# Main storage directory. You'll need disk space to dump the WHAM mixtures and the wsj0 wav
# files if you start from sphere files.
storage_dir=./datasets

librispeech_dir=$storage_dir/LibriSpeech #$storage_dir/LibriSpeech
rir_dir=$storage_dir/rir_data #$storage_dir/rir_data
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

# Data
sample_rate=16000


. utils/parse_options.sh

sr_string=$(($sample_rate/1000))
suffix=wav${sr_string}k
dumpdir=data/$suffix  # directory to put generated json file



# Check if sox is installed
if ! [[ -x "$(command -v sox)" ]] ; then
  echo "This recipe requires SoX, Install sox with `conda install -c conda-forge sox`. Exiting."
  exit 1
fi

# Install pysndfx if not instaled
if not python -c "import pysndfx" &> /dev/null; then
  echo 'This recipe requires pysndfx. Installing requirements.'
  $python_path -m pip install -r requirements.txt
fi

if [[ $stage -le  0 ]]; then
  echo "Stage 0: Downloading required Datasets"

  if ! test -e $librispeech_dir/train-clean-360; then
    echo "Downloading LibriSpeech/train-clean-360 into $storage_dir"
    wget -c --tries=0 --read-timeout=20 http://www.openslr.org/resources/12/train-clean-360.tar.gz -P $storage_dir
	  tar -xzf $storage_dir/train-clean-360.tar.gz -C $storage_dir
	  rm -rf $storage_dir/train-clean-360.tar.gz
	fi

  if ! test -e $librispeech_dir/dev-clean; then
    echo "Downloading LibriSpeech/dev-clean into $storage_dir"
	  wget -c --tries=0 --read-timeout=20 http://www.openslr.org/resources/12/dev-clean.tar.gz -P $storage_dir
	  tar -xzf $storage_dir/dev-clean.tar.gz -C $storage_dir
	  rm -rf $storage_dir/dev-clean.tar.gz
	fi

  if ! test -e $rir_dir; then
    echo "Downloading FUSS rir data"
    wget -c --tries=0 --read-timeout=20 https://zenodo.org/record/3743844/files/FUSS_rir_data.tar.gz -P $storage_dir
	  tar -xzf $storage_dir/FUSS_rir_data.tar.gz -C $storage_dir
	  rm -rf $storage_dir/FUSS_rir_data.tar.gz
	fi

fi

if [[ $stage -le  1 ]]; then
  echo "Stage 1: parsing the datasets"
	for librispeech_split in train-clean-360 dev-clean ; do
    python local/parse_data.py --input_dir $librispeech_dir/$librispeech_split --output_json $dumpdir/clean/${librispeech_split}.json --regex **/*.flac
  done
  for rir_split in train validation ; do
    python local/parse_data.py --input_dir $rir_dir/$rir_split --output_json $dumpdir/rirs/${rir_split}.json --regex **/*.wav
  done
fi

# Generate a random ID for the run if no tag is specified
uuid=$($python_path -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
if [[ -z ${tag} ]]; then
	tag=${sr_string}k_${uuid}
fi
expdir=exp/train_demask_${tag}
mkdir -p $expdir && echo $uuid >> $expdir/run_uuid.txt
echo "Results from the following experiment will be stored in $expdir"

if [[ $stage -le 2 ]]; then
  echo "Stage 3: Training"
  mkdir -p logs
  CUDA_VISIBLE_DEVICES=$id $python_path train.py \
		--clean_speech_train $dumpdir/clean/train-clean-360.json \
		--clean_speech_valid $dumpdir/clean/dev-clean.json \
	  --rir_train $dumpdir/rirs/train.json \
	  --rir_valid $dumpdir/rirs/validation.json \
		--fs $sample_rate \
		--exp_dir ${expdir}/ | tee logs/train_${tag}.log
	cp logs/train_${tag}.log $expdir/train.log

	# Get ready to publish
	mkdir -p $expdir/publish_dir
	echo "DeMask" > $expdir/publish_dir/recipe_name.txt
fi
