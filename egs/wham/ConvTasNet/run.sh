#!/bin/bash

# * If you have mixture wsj0 audio, modify `data` to your path that including tr, cv and tt.
# * If you jsut have origin sphere format wsj0 , modify `wsj0_origin` to your path and
# modify `wsj0_wav` to path that put output wav format wsj0, then read and run stage 1 part.
# After that, modify `data` and run from stage 2.

base=/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/mpariente/
sphere_dir=  # Directory containing sphere files
wsj0_wav_dir=${base}/DATA/wsj0_wav/ # Directory where to save wsj0 wav files (Need disk space)
wham_wav_dir=${base}/wham_recipe_results/  # # Directory where to save WHAM wav files (Need disk space)

stage=-1
tag=""
python_path=${base}/asteroid_conda/miniconda3/bin/python
id=

data_dir=data  # Local data directory (No disk space needed)

use_cuda=0
sample_rate=8000
mask_nonlinear=relu
batch_size=3
num_workers=3
optimizer=adam
lr=0.001
checkpoint=0
epochs=100
continue_from=
model_path=final.pth.tar
print_freq=1000


task=sep_clean
sample_rate=8000
mode=min
nondefault_src=


. utils/parse_options.sh || exit 1;

mkdir -p logs


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


if [[ $stage -le  0 ]]; then
	echo "Stage 0: Converting sphere files to wav files"
  . local/convert_sphere2wav.sh --sphere_dir $sphere_dir --wav_dir $wsj0_wav_dir
fi


if [[ $stage -le  1 ]]; then
	echo "Stage 1: Generating 8k and 16k WHAM dataset"
  . local/prepare_data.sh --wav_dir $wsj0_wav_dir --out_dir $wham_wav_dir --python_path $python_path
fi


if [[ $stage -le  2 ]]; then
	# Make json directories with all modes and sampling rates
	echo "Stage 2: Generating json files including wav path and duration"
	for sr_string in 8 16; do
		for mode in min max; do
			tmp_dumpdir=data/wav${sr_string}k/$mode
			echo "Generating json files in $tmp_dumpdir"
			[[ ! -d $tmp_dumpdir ]] && mkdir -p $tmp_dumpdir
			local_wham_dir=$wham_wav_dir/wav${sr_string}k/$mode/
      $python_path local/preprocess_wham.py --in-dir $local_wham_dir --out-dir $tmp_dumpdir
    done
  done
fi

sr_string=$(($sample_rate/1000))
suffix=wav${sr_string}k/$mode
dumpdir=data/$suffix  # directory to put generated json file

train_dir=$dumpdir/tr
valid_dir=$dumpdir/cv
evaluate_dir=$dumpdir/tt

uuid=$($python_path -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
if [[ -z ${tag} ]]; then
	tag=${task}_${sr_string}k${mode}_${uuid}
fi
expdir=exp/train_convtasnet_${tag}
mkdir -p $expdir && echo $uuid >> $expdir/run_uuid.txt
echo "Results from the following experiment will be stored in $expdir"

if [[ $stage -le 3 ]]; then
  echo "Stage 3: Training"
  CUDA_VISIBLE_DEVICES=$id $python_path train.py \
  --train_dir $train_dir \
  --valid_dir $valid_dir \
  --use_cuda $use_cuda \
  --sample_rate $sample_rate \
  --mask_act $mask_nonlinear \
  --use_cuda $use_cuda \
  --epochs $epochs \
  --batch_size $batch_size \
  --num_workers $num_workers \
  --lr $lr \
  --save_folder ${expdir} \
  --checkpoint $checkpoint \
  --continue_from "$continue_from" \
  --model_path $model_path \
  --print_freq $print_freq | tee logs/train_${tag}.log
fi
#  --optimizer $optimizer \


if [[ $stage -le 4 ]]; then
	echo "Stage 4 : Evaluation"
	CUDA_VISIBLE_DEVICES=$id $python_path eval.py
fi