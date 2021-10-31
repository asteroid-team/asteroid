#!/bin/bash

# Exit on error
set -e
set -o pipefail

# The root directory containing CHiME3
storage_dir=

# Directory containing the pretrained model
exp_dir=
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

eval_use_gpu=0
# Need to --compute_wer 1 --eval_mode max to be sure the user knows all the metrics
# are for the all mode.
compute_wer=1

# Choice for the ASR model whether trained on clean or noisy data. One of clean or noisy
asr_type=noisy

. utils/parse_options.sh

test_dir=data/test

if [[ $stage -le  0 ]]; then
	echo "Stage 0: Generating CHiME-4 dataset"
  $python_path local/create_metadata.py --chime3_dir $storage_dir/CHiME3/
fi

if [[ $stage -le 1 ]]; then
  echo "Stage 2 : Evaluation"
  echo "Results from the following experiment will be stored in $exp_dir/chime4/$asr_type"

	if [[ $compute_wer -eq 1 ]]; then

    # Install espnet if not instaled
    if ! python -c "import espnet" &> /dev/null; then
        echo 'This recipe requires espnet. Installing requirements.'
        $python_path -m pip install espnet_model_zoo
        $python_path -m pip install jiwer
        $python_path -m pip install tabulate
    fi
  fi

  $python_path eval.py \
    --exp_dir $exp_dir \
    --test_dir $test_dir \
  	--use_gpu $eval_use_gpu \
  	--compute_wer $compute_wer \
  	--asr_type $asr_type
fi
