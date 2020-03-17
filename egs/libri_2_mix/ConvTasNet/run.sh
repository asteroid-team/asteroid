#!/bin/bash
set -e  # Exit on error
# Main storage directory. You'll need disk space to dump the raw LibriSpeech flac files
# and generate the sources and mixtures
storage_dir=/srv/storage/talc@talc-data.nancy/multispeech/calcul/users/jcosentino/DATA/

# If you haven't downloaded and extracted the LibriSpeech dataset start from stage 1

# If you already have dowloaded and extracted LibriSpeech,
# specify the path to the directory here and start from stage 2
LibriSpeech_dir=${storage_dir}

# After running the recipe a first time, you can run it from stage 3 directly to train new models.

# Path to the python you'll use for the experiment. Defaults to the current python
# You can run ./utils/prepare_python_env.sh to create a suitable python environment, paste the output here.
#python_path=${storage_dir}/asteroid_conda/miniconda3/bin/python
python_path=python

# Example usage
# ./run.sh --stage 3 --tag my_tag --task sep_noisy --id 0,1

# General
stage=3  # Controls from which stage to start
tag=""  # Controls the directory name associated to the experiment
# You can ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES)
id=0,1,2,3

# Data
data_dir=data  # Local data directory (No disk space needed)
task=sep_clean  # Specify the task here (sep_clean, sep_noisy, enh_single, enh_both)
sample_rate=16000


# Training
batch_size=8
num_workers=8
#optimizer=adam
lr=0.001
epochs=200

# Architecture
n_blocks=6
n_repeats=2
mask_nonlinear=relu

# Evaluation
eval_use_gpu=1


. utils/parse_options.sh


if [[ $stage -le  -1 ]]; then
	echo "Stage -1: Creating python environment to run this"
	if [[ -x "${python_path}" ]]
	then
		echo "The provided python path is executable, don't proceed to installation."
	else
	  . utils/prepare_python_env.sh --install_dir $python_path --asteroid_root ../../..
	  echo "Miniconda3 install can be found at $python_path"
	  python_path=${python_path}/miniconda3/bin/python
	  echo -e "\n To use this python version for the next experiments, change"
	  echo -e "python_path=$python_path at the beginning of the file \n"
	fi
fi



if [[ $stage -le  1 ]]; then
	echo "Stage 1: Downloading LibriSpeech and Generating 8k and 16k Libri2mix dataset"
  . local/prepare_data.sh --out_dir $storage_dir --in_dir $storage_dir --python_path $python_path
fi

if [[ $stage -le  2 ]]; then
	echo "Stage 1: Generating 8k and 16k Libri2mix dataset"
  $python_path local/create_libri2mix_from_scratch.py --in_dir $LibriSpeech_dir --out_dir $storage_dir
fi



# Generate a random ID for the run if no tag is specified
uuid=$($python_path -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
if [[ -z ${tag} ]]; then
	tag=${task}_${sr_string}k${mode}_${uuid}
fi
expdir=exp/train_convtasnet_${tag}
mkdir -p $expdir && echo $uuid >> $expdir/run_uuid.txt
echo "Results from the following experiment will be stored in $expdir"


if [[ $stage -le 3 ]]; then
  echo "Stage 3: Training"
  mkdir -p logs
  CUDA_VISIBLE_DEVICES=$id $python_path train.py --exp_dir exp/tmp \
  --task $task \
  --sample_rate $sample_rate \
  --lr $lr \
  --epochs $epochs \
  --batch_size $batch_size \
  --num_workers $num_workers \
  --mask_act $mask_nonlinear \
  --n_blocks $n_blocks \
  --n_repeats $n_repeats \
  --exp_dir ${expdir}/ | tee logs/train_${tag}.log
fi


if [[ $stage -le 4 ]]; then
	echo "Stage 4 : Evaluation"
	CUDA_VISIBLE_DEVICES=$id $python_path eval.py \
	--task $task \
	--test_dir exp/tmp \
	--use_gpu $eval_use_gpu \
	--exp_dir exp/tmp
fi
