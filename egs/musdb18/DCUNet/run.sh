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
stage=4  # Controls from which stage to start
tag=""  # Controls the directory name associated to the experiment
# You can ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES)
id=0 #$CUDA_VISIBLE_DEVICES this does not work, so I put 0,1 manually.


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

eval_use_gpu=1 # 1 meaning true

. utils/parse_options.sh

#DCUNET:

#root=../../../../toydata #choose your aligned dataset here
root=../../../../data_extractednaive4_wmasks/

# possible architectures: "DCUNet-10", "DCUNet-16", "DCUNet-20", "Large-DCUNet-20"
architecture="DCUNet-10"

if [[ $stage -le  0 ]]; then
  echo "Stage 0: Converting sphere files to wav files"
  . local/convert_sphere2wav.sh --sphere_dir $sphere_dir --wav_dir $wsj0_wav_dir
fi

if [[ $stage -le  1 ]]; then
	echo "Stage 1: Generating 8k and 16k WHAM dataset"
  . local/prepare_data.sh --wav_dir $wsj0_wav_dir --out_dir $wham_wav_dir --python_path $python_path
fi

if [[ $stage -le  2 ]]; then
	# Make json directories with min/max modes and sampling rates
	echo "Stage 2: Generating json files including wav path and duration"
	for sr_string in 8 16; do
		for mode_option in min max; do
			tmp_dumpdir=data/wav${sr_string}k/$mode_option
			echo "Generating json files in $tmp_dumpdir"
			[[ ! -d $tmp_dumpdir ]] && mkdir -p $tmp_dumpdir
			local_wham_dir=$wham_wav_dir/wav${sr_string}k/$mode_option/
      $python_path local/preprocess_wham.py --in_dir $local_wham_dir --out_dir $tmp_dumpdir
    done
  done
fi

# Generate a random ID for the run if no tag is specified
uuid=$($python_path -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
if [[ -z ${tag} ]]; then
	tag=${task}_${sr_string}k${mode}_${uuid}
fi
#expdir=exp/train_DCUNET_${tag}
expdir=exp/train_DCUNET__k_9a0f7bd2
mkdir -p $expdir && echo $uuid >> $expdir/run_uuid.txt
echo "Results from the following experiment will be stored in $expdir"

if [[ $stage -le 3 ]]; then
  echo "Stage 3: Training"
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

if [[ $stage -le 4 ]]; then
	echo "Stage 4 : Evaluation"
	CUDA_VISIBLE_DEVICES=$id $python_path eval.py \
		--use_gpu $eval_use_gpu \
		--exp_dir ${expdir} | tee logs/eval_${tag}.log
	cp logs/eval_${tag}.log $expdir/eval.log
fi
