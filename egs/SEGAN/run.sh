#!/bin/bash
set -e  # Exit on error

# If you haven't generated LibriMix start from stage 0
# Main storage directory. You'll need disk space to store LibriSpeech, WHAM noises
# and LibriMix. This is about 500 Gb
storage_dir=

# After running the recipe a first time, you can run it from stage 3 directly to train new models.

# Path to the python you'll use for the experiment. Defaults to the current python
# You can run ./utils/prepare_python_env.sh to create a suitable python environment, paste the output here.
python_path=python

# Example usage
# ./run.sh --stage 3 --tag my_tag --task sep_noisy --id 0,1

# General
stage=1  # Controls from which stage to start
tag=""  # Controls the directory name associated to the experiment
# You can ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES)
test_dir=data/wav16k/min/test/
. ../wham/ConvTasNet/utils/parse_options.sh


if [[ $stage -le  0 ]]; then
	echo "Stage 0: Generating Librimix dataset"
  . ../librimix/ConvTasNet/local/prepare_data.sh --storage_dir $storage_dir --n_src 2
fi

# Generate a random ID for the run if no tag is specified
uuid=$($python_path -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
if [[ -z ${tag} ]]; then
	tag=${uuid}
fi

expdir=exp/train_SEGAN_${tag}
mkdir -p $expdir && echo $uuid >> $expdir/run_uuid.txt
echo "Results from the following experiment will be stored in $expdir"


if [[ $stage -le 1 ]]; then
  echo "Stage 1: Training"
  mkdir -p logs
  $python_path train.py  \
    --exp_dir $expdir/ | tee logs/train_${tag}.log \
	cp logs/train_${tag}.log $expdir/train.log
fi

if [[ $stage -le 2 ]]; then
	echo "Stage 2 : Evaluation"
  $python_path eval.py --test_dir $test_dir \
    --exp_dir ${expdir} | tee logs/eval_${tag}.log
	cp logs/eval_${tag}.log $expdir/eval.log
fi
