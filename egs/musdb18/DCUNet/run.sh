#!/bin/bash



# Exit on error
set -e
set -o pipefail

# Path to the python you'll use for the experiment. Defaults to the current python
# You can run ./utils/prepare_python_env.sh to create a suitable python environment, paste the output here.
python_path=python3

# Example usage
# ./run.sh --stage 3 --tag my_tag --task sep_noisy --id 0,1

# General

stage=1  # Controls from which stage to start: 1 to start with training, 2 to evaluate.

#choose your MUSDB dataset here. MUSDB full dataset can be requested from https://zenodo.org/record/1117372#.X9OXs-lKjOQ
#if you would like to download a preview to test the model instead, put stage=0 to download MUSDB to the root directory.

root=../../../../data_extractednaive4_wmasks/

tag=""  # Controls the directory name associated to the experiment
# You can ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES)
id=0,1 #$CUDA_VISIBLE_DEVICES this does not work, so I put 0,1 manually.

# Data
sample_rate=44100

# Training
batch_size=6
num_workers=6
optimizer=adam
lr=0.001
epochs=200
weight_decay=0.00001

# Architecture config
stft_kernel_size=4096
stft_stride=1024
chunk_size=100
hop_size=50

eval_use_gpu=1 # 1 meaning true, 0 false.

. utils/parse_options.sh

# possible architectures: "DCUNet-10", "DCUNet-16", "DCUNet-20", "Large-DCUNet-20"
architecture="DCUNet-10"

# Generate a random ID for the run if no tag is specified
uuid=$($python_path -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
if [[ -z ${tag} ]]; then
	tag=${task}_${sr_string}k${mode}_${uuid}
fi

#expdir=exp/train_DCUNET_${tag}
expdir=exp/train_DCUNET__k_9a0f7bd2

mkdir -p $expdir && echo $uuid >> $expdir/run_uuid.txt
echo "Results from the following experiment will be stored in $expdir"

#sample-url: https://zenodo.org/api/files/1ff52183-071a-4a59-923f-7a31c4762d43/MUSDB18-7-STEMS.zip
if [[ $stage -le 0 ]]; then
  wget  -O "${root}zipped" "https://zenodo.org/api/files/1ff52183-071a-4a59-923f-7a31c4762d43/MUSDB18-7-STEMS.zip"
  unzip "${root}zipped" -d $root
fi

if [[ $stage -le 1 ]]; then
  echo "Stage 1: Training"
  mkdir -p logs

  CUDA_VISIBLE_DEVICES=$id $python_path train.py \
		--root $root\
		--architecture $architecture\
		--sample_rate $sample_rate \
		--lr $lr \
		--epochs $epochs \
		--batch_size $batch_size \
		--num_workers $num_workers \
		--optimizer $optimizer \
		--weight_decay $weight_decay \
		--stft_kernel_size $stft_kernel_size \
		--stft_stride $stft_stride \
		--exp_dir ${expdir}/ | tee logs/train_${tag}.log
	cp logs/train_${tag}.log $expdir/train.log

	# Get ready to publish
	mkdir -p $expdir/publish_dir
	echo "musdb18/DCUNet" > $expdir/publish_dir/recipe_name.txt
fi

if [[ $stage -le 2 ]]; then
	echo "Stage 2 : Evaluation"
	CUDA_VISIBLE_DEVICES=$id $python_path eval.py \
		--use_gpu $eval_use_gpu \
		--exp_dir ${expdir} | tee logs/eval_${tag}.log
	cp logs/eval_${tag}.log $expdir/eval.log
fi
