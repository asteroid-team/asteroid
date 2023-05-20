#!/bin/bash

echo "pretrained model can be found at: https://huggingface.co/JunzheJosephZhu/MultiDecoderDPRNN"

# Exit on error
set -e
set -o pipefail

# Main storage directory. You'll need disk space to dump the WHAM mixtures and the wsj0 wav
# files if you start from sphere files.
storage_dir=
# If you start from the sphere files, specify the path to the directory and start from stage 0
sphere_dir=  # Directory containing sphere files
# If you already have wsj0 wav files (converted from sphere format).
wsj0_wav_dir=
# If you already have the wsj0-2mix and wsj0-3mix mixtures, specify the path to the common directory
# and start from stage 2.
wsj0mix_wav_dir=/ws/ifp-10_3/hasegawa/junzhez2/asteroid/dataset

# After running the recipe a first time, you can run it from stage 3 directly to train new models.

# Path to the python you'll use for the experiment. Defaults to the current python
# You can run ./utils/prepare_python_env.sh to create a suitable python environment, paste the output here.
python_path=python

# Example usage
# ./run.sh --stage 3 --tag my_tag --loss_alpha 0.1 --id 0,1

# General
stage=4  # Controls from which stage to start
tag=""  # Controls the directory name associated to the experiment
id=$CUDA_VISIBLE_DEVICES
# You can ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES). To do so, uncomment the line below.
# id=(1 2 3 4)


# Data
#data_dir=data  # Local data directory (No disk space needed)
sample_rate=8000
mode=min
n_srcs=(2 3 4 5)

# Training
batch_size=2
num_workers=2
optimizer=adam
lr=0.001
weight_decay=0.0
epochs=200
lambda=0.05
resume_from=

# Evaluation
eval_use_gpu=1

. utils/parse_options.sh

sr_string=$(($sample_rate/1000))
suffix={}speakers/wav${sr_string}k/$mode
dumpdir=data/$suffix  # directory to put generated json file

train_dir=$dumpdir/tr
valid_dir=$dumpdir/cv
test_dir=$dumpdir/tt

if [[ $stage -le  0 ]]; then
  echo "Stage 0: Converting sphere files to wav files"
  . local/convert_sphere2wav.sh --sphere_dir $sphere_dir --wav_dir $wsj0_wav_dir
fi

if [[ $stage -le  1 ]]; then
	echo "Stage 1 : Downloading wsj0-mix mixing scripts"
	# Link + WHAM is ok for 2 source.
	# wget https://www.merl.com/demos/deep-clustering/create-speaker-mixtures.zip -O ./local/
	wget https://github.com/JunzheJosephZhu/MDDPRNN-deprecated/raw/master/create-speaker-mixtures-2345.zip -P ./local
	unzip ./local/create-speaker-mixtures-2345.zip -d ./local/create-speaker-mixtures-2345
	mv ./local/create-speaker-mixtures-2345.zip ./local/create-speaker-mixtures-2345

	echo "You need to generate the wsj0-mix dataset using the official MATLAB
			  scripts (already downloaded into ./local/create-speaker-mixtures-2345).
			  If you don't have Matlab, you can use Octavve and replace
				all mkdir(...) in create_wav_2speakers.m with system(['mkdir -p '...]).
				Specify wsj0mix_wav_dir and start from stage 2 when the mixtures have been generated.
				Exiting now."
	exit 1
fi

if [[ $stage -le  2 ]]; then
	# Make json directories with min/max modes and sampling rates
	echo "Stage 2: Generating json files including wav path and duration"
	for tmp_nsrc in "${n_srcs[@]}"; do
		tmp_dumpdir=$wsj0mix_wav_dir/${tmp_nsrc}speakers/wav8k/min
		echo "Generating json files in $tmp_dumpdir"
		[[ ! -d $tmp_dumpdir ]] && mkdir -p $tmp_dumpdir
		local_wsj_dir=$wsj0mix_wav_dir/${tmp_nsrc}speakers/wav8k/min/
		$python_path local/preprocess_wsj0mix.py --in_dir $local_wsj_dir \
																					--n_src $tmp_nsrc \
																					--out_dir $tmp_dumpdir
	done
fi

# Generate a random ID for the run if no tag is specified
uuid=$($python_path -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
if [[ -z ${tag} ]]; then
	tag=$( IFS=$''; echo "${n_srcs[*]}" )sep_${sr_string}k${mode}_${uuid}
fi
expdir=exp/tmp_${tag}
mkdir -p $expdir && echo $uuid >> $expdir/run_uuid.txt
mkdir -p logs
echo "Results from the following experiment will be stored in $expdir"

if [[ $stage -le 3 ]]; then
  echo "Stage 3: Training"
  echo "visible cuda devices are ${id[*]}"
  mkdir -p logs
  CUDA_VISIBLE_DEVICES=$id $python_path train.py \
		--train_dir $train_dir \
		--valid_dir $valid_dir \
		--sample_rate $sample_rate \
		--optimizer $optimizer \
		--lr $lr \
		--weight_decay $weight_decay \
		--epochs $epochs \
		--batch_size $batch_size \
		--lambda $lambda \
		--num_workers $num_workers \
		--exp_dir ${expdir}/ | tee logs/train_${tag}.log
		--resume_from $resume_from
	cp logs/train_${tag}.log $expdir/train.log
fi

if [[ $stage -le 4 ]]; then
	echo "Stage 4 : Evaluation"
	echo "If you want to change n_srcs, please change the config file"
	CUDA_VISIBLE_DEVICES=$id $python_path eval.py \
		--test_dir $test_dir \
		--use_gpu $eval_use_gpu \
		--exp_dir ${expdir} | tee logs/eval_${tag}.log
	cp logs/eval_${tag}.log $expdir/eval.log
fi
