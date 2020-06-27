#!/bin/bash
set -e  # Exit on error

# path to root of MiniLibriMix
mini_librimix_storage_dir=/home/sam/Projects/lhotse/examples/librimix/
# After running the recipe a first time, you can run it from stage 2 directly to train new models.

# Path to the python you'll use for the experiment. Defaults to the current python
# You can run ./utils/prepare_python_env.sh to create a suitable python environment, paste the output here.
python_path=python

# Example usage
# ./run.sh --stage 3 --tag my_tag --task sep_noisy --id 0,1

# General
stage=2  # Controls from which stage to start
tag="test"  # Controls the directory name associated to the experiment
# You can ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES)
id=$CUDA_VISIBLE_DEVICES

# Data
task=sep_clean  # Specify the task here (sep_clean, sep_noisy, enh_single, enh_both)
sample_rate=8000
mode=min
#nondefault_src= unsupported for.

# Training
batch_size=6
num_workers=6
optimizer=adam
lr=0.001
epochs=200
weight_decay=0.00001

# Architecture config
kernel_size=16
stride=8
chunk_size=100
hop_size=50
segment=3

# Evaluation
eval_use_gpu=1

. utils/parse_options.sh

sr_string=$(($sample_rate/1000))
suffix=wav${sr_string}k/$mode
dumpdir=data/$suffix  # directory to put generated json file

train_dir=$dumpdir/tr
valid_dir=$dumpdir/cv
test_dir=$dumpdir/tt

if [[ -z $mini_librimix_storage_dir ]]; then
  echo "Specify a path for MiniLibriMix, exiting...."
  exit
fi

if [[ $stage -le  0 ]]; then
  echo "Stage 0: Downloading MiniLibriMix"
  if [ ! -d $mini_librimix_storage_dir ]; then
  wget https://zenodo.org/record/3871592/files/MiniLibriMix.zip
  unzip MiniLibriMix.zip
  fi
fi

if [[ $stage -le  1 ]]; then
	echo "Stage 1: Parsing examples with lhotse"
  ./local/librimix.sh $mini_librimix_storage_dir ./data $segment
fi


# Generate a random ID for the run if no tag is specified
uuid=$($python_path -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
if [[ -z ${tag} ]]; then
	tag=${task}_${sr_string}k${mode}_${uuid}
fi
expdir=exp/train_dprnn_${tag}
mkdir -p $expdir && echo $uuid >> $expdir/run_uuid.txt
echo "Results from the following experiment will be stored in $expdir"

if [[ $stage -le 2 ]]; then
  echo "Stage 3: Training"
  mkdir -p logs
  CUDA_VISIBLE_DEVICES=$id $python_path train.py \
		--train_dir $train_dir \
		--valid_dir $valid_dir \
		--task $task \
		--sample_rate $sample_rate \
		--lr $lr \
		--epochs $epochs \
		--batch_size $batch_size \
		--num_workers $num_workers \
		--optimizer $optimizer \
		--weight_decay $weight_decay \
		--chunk_size $chunk_size \
		--hop_size $hop_size \
		--exp_dir ${expdir}/ | tee logs/train_${tag}.log
	cp logs/train_${tag}.log $expdir/train.log

	# Get ready to publish
	mkdir -p $expdir/publish_dir
	echo "wham/DPRNN" > $expdir/publish_dir/recipe_name.txt
fi

if [[ $stage -le 3 ]]; then
	echo "Stage 4 : Evaluation"
	CUDA_VISIBLE_DEVICES=$id $python_path eval.py \
		--task $task \
		--test_dir $test_dir \
		--use_gpu $eval_use_gpu \
		--exp_dir ${expdir} | tee logs/eval_${tag}.log
	cp logs/eval_${tag}.log $expdir/eval.log
fi
