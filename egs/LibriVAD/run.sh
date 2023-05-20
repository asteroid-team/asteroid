#!/bin/bash

# Exit on error
set -e
set -o pipefail

# If you haven't generated Libri_VAD start from stage 0
storage_dir=

# After running the recipe a first time, you can run it from stage 3 directly to train new models.

# Path to the python you'll use for the experiment. Defaults to the current python
# You can run ./utils/prepare_python_env.sh to create a suitable python environment, paste the output here.
python_path=python

# Example usage
# ./run.sh --stage 0 --tag my_tag --id 0,1

# General
stage=0  # Controls from which stage to start
tag=""  # Controls the directory name associated to the experiment
# You can ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES)
id=$CUDA_VISIBLE_DEVICES
out_dir=vad_result # Controls the directory name associated to the evaluation results inside the experiment directory

. utils/parse_options.sh


if [[ $stage -le  0 ]]; then
	echo "Stage 0: Generating VAD dataset"
  . local/generate_librivad.sh --storage_dir $storage_dir
fi

if [[ $stage -le  1 ]]; then
	echo "Stage 1: Generating json files"
  . local/prepare_data.sh
fi

# Generate a random ID for the run if no tag is specified
uuid=$($python_path -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
if [[ -z ${tag} ]]; then
	tag=${uuid}
fi

expdir=exp/train_convtasnet_${tag}
mkdir -p $expdir && echo $uuid >> $expdir/run_uuid.txt
echo "Results from the following experiment will be stored in $expdir"


if [[ $stage -le 2 ]]; then
  echo "Stage 2: Training"
  mkdir -p logs
  CUDA_VISIBLE_DEVICES=$id $python_path train.py --exp_dir $expdir | tee logs/train_${tag}.log
	cp logs/train_${tag}.log $expdir/train.log
fi

md_path=data/test.json

if [[ $stage -le 3 ]]; then
	echo "Stage 3 : Evaluation"
  $python_path eval.py \
    --exp_dir $expdir \
  	--out_dir $out_dir \
  	--md_path $md_path | tee logs/eval_${tag}.log

	cp logs/eval_${tag}.log $expdir/eval.log
fi
