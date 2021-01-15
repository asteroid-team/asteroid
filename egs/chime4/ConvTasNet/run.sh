#!/bin/bash

# Exit on error
set -e
set -o pipefail

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
stage=0  # Controls from which stage to start
tag=""  # Controls the directory name associated to the experiment
# You can ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES)

eval_use_gpu=1
# Need to --compute_wer 1 --eval_mode max to be sure the user knows all the metrics
# are for the all mode.
compute_wer=0
eval_mode=

. utils/parse_options.sh


if [[ $stage -le  0 ]]; then
	echo "Stage 0: Generating CHiME-4 dataset"
  . local/create_metadata.py --chime3_dir $storage_dir/CHiME3
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
	echo "Stage 2 : Evaluation"

	if [[ $compute_wer -eq 1 ]]; then
	  if [[ $eval_mode != "max" ]]; then
	    echo "Cannot compute WER without max mode. Start again with --stage 2 --compute_wer 1 --eval_mode max"
	    exit 1
	  fi

    # Install espnet if not instaled
    if ! python -c "import espnet" &> /dev/null; then
        echo 'This recipe requires espnet. Installing requirements.'
        $python_path -m pip install espnet_model_zoo
        $python_path -m pip install jiwer
        $python_path -m pip install tabulate
    fi
  fi

  $python_path eval.py \
    --exp_dir $expdir \
    --test_dir $test_dir \
  	--use_gpu $eval_use_gpu \
  	--compute_wer $compute_wer | tee logs/eval_${tag}.log

	cp logs/eval_${tag}.log $expdir/eval.log
fi
