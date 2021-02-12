#!/bin/bash

# Exit on error
set -e
set -o pipefail

# Main storage directory where dataset will be stored
storage_dir=$(readlink -m ./datasets)
librispeech_dir=$storage_dir/LibriSpeech
noise_dir=$storage_dir/Nonspeech
# After running the recipe a first time, you can run it from stage 3 directly to train new models.

# Path to the python you'll use for the experiment. Defaults to the current python
# You can run ./utils/prepare_python_env.sh to create a suitable python environment, paste the output here.
python_path=python

# Example usage
# ./run.sh --stage 3 --tag my_tag --id 0,1

# General
stage=0  # Controls from which stage to start
tag=""  # Controls the directory name associated to the experiment
# You can ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES)
id=0
eval_use_gpu=1

# Dataset option
dataset_type=adhoc
samplerate=16000

. utils/parse_options.sh

dumpdir=data/$suffix  # directory to put generated json file

# check if gpuRIR installed
if ! ( pip list | grep -F gpuRIR ); then
  echo 'This recipe requires gpuRIR. Please install gpuRIR.'
  exit
fi

if [[ $stage -le  0 ]]; then
  echo "Stage 0: Downloading required Datasets"

  if ! test -e $librispeech_dir/train-clean-100; then
    echo "Downloading LibriSpeech/train-clean-100 into $storage_dir"
    wget -c --tries=0 --read-timeout=20 http://www.openslr.org/resources/12/train-clean-100.tar.gz -P $storage_dir
	  tar -xzf $storage_dir/train-clean-100.tar.gz -C $storage_dir
	  rm -rf $storage_dir/train-clean-100.tar.gz
	fi

  if ! test -e $librispeech_dir/dev-clean; then
    echo "Downloading LibriSpeech/dev-clean into $storage_dir"
	  wget -c --tries=0 --read-timeout=20 http://www.openslr.org/resources/12/dev-clean.tar.gz -P $storage_dir
	  tar -xzf $storage_dir/dev-clean.tar.gz -C $storage_dir
	  rm -rf $storage_dir/dev-clean.tar.gz
	fi

  if ! test -e $librispeech_dir/test-clean; then
    echo "Downloading LibriSpeech/test-clean into $storage_dir"
	  wget -c --tries=0 --read-timeout=20 http://www.openslr.org/resources/12/test-clean.tar.gz -P $storage_dir
	  tar -xzf $storage_dir/test-clean.tar.gz -C $storage_dir
	  rm -rf $storage_dir/test-clean.tar.gz
	fi

  if ! test -e $storage_dir/Nonspeech; then
       echo "Downloading Noises into $storage_dir"
	  wget -c --tries=0 --read-timeout=20 http://web.cse.ohio-state.edu/pnl/corpus/HuNonspeech/Nonspeech.zip -P $storage_dir
	  unzip $storage_dir/Nonspeech.zip -d $storage_dir
	  rm -rf $storage_dir/Nonspeech.zip
  fi

fi

if [[ $stage -le  1 ]]; then
  echo "Stage 1: Creating Synthetic Datasets"
  git clone https://github.com/yluo42/TAC ./local/TAC
  cd local/TAC/data
  $python_path create_dataset.py \
                --output-path=$storage_dir \
		            --dataset=$dataset_type \
		            --libri-path=$librispeech_dir \
		            --noise-path=$noise_dir
  cd ../../../
fi

if [[ $stage -le 2 ]]; then
  echo "Parsing dataset to json to speed up subsequent experiments"
  for split in train validation test; do
      $python_path ./local/parse_data.py --in_dir $storage_dir/MC_Libri_${dataset_type}/$split --out_json $dumpdir/${split}.json
  done
fi

# Generate a random ID for the run if no tag is specified
uuid=$($python_path -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
if [[ -z ${tag} ]]; then
	tag=${uuid}
fi
expdir=exp/train_TAC_${tag}
mkdir -p $expdir && echo $uuid >> $expdir/run_uuid.txt
echo "Results from the following experiment will be stored in $expdir"

if [[ $stage -le 3 ]]; then
  echo "Stage 3: Training"
  mkdir -p logs
  CUDA_VISIBLE_DEVICES=$id $python_path train.py --sample_rate $samplerate --exp_dir ${expdir} | tee logs/train_${tag}.log
	cp logs/train_${tag}.log $expdir/train.log

	# Get ready to publish
	mkdir -p $expdir/publish_dir
	echo "TAC/TAC" > $expdir/publish_dir/recipe_name.txt
fi


if [[ $stage -le 4 ]]; then
	echo "Stage 4 : Evaluation"
	CUDA_VISIBLE_DEVICES=$id $python_path eval.py --test_json $dumpdir/test.json \
		--use_gpu $eval_use_gpu \
		--exp_dir ${expdir} | tee logs/eval_${tag}.log
	cp logs/eval_${tag}.log $expdir/eval.log
fi
