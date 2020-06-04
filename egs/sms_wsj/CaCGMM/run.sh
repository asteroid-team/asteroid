#!/bin/bash


storage_dir=/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/mpariente/ # Main storage directory (Need disk space)
wsj_dir=/net/fastdb/wsj # WSJ DIR (if wsj0 and wsj1 have been merged)

# Following wsj_dirs have to be defined, if wsj0 and wsj1 have not been merged
wsj0_dir=$wsj_dir
wsj1_dir=$wsj_dir

num_jobs=$(nproc --all)

stage=-1

data_dir=data  # Local data directory (Not much disk space required)
python_path=${storage_dir}/asteroid_conda/miniconda3/bin/python
. utils/parse_options.sh || exit 1;

sms_wsj=${storage_dir}/DATA/sms_wsj/ # Directory where to save SMS-WSJ wav files
anaconda_path=${storage_dir}/asteroid_conda/

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [[ $stage -le  -1 ]]; then
	echo "Stage -1: Creating python environment to run this"
	if [[ -x "${python_path}" ]]
	then
		echo "The provided python path is executable, don't proceed to installation."
	else
		if [[ ! -x "${anaconda_path}" ]]; then
	  		. utils/prepare_python_env.sh --install_dir $anaconda_path --asteroid_root ../../..
	  	fi
	  	python_path=${anaconda_path}/miniconda3/bin/python
		echo "Miniconda3 install can be found at $python_path"
		echo -e "\n To use this python version for the next experiments, change"
		echo -e "python_path=$python_path at the beginning of the file \n"
	fi
fi


if [[ $stage -le  0 ]]; then
	echo "Stage 0: Cloning and installing SMS-WSJ repository"
	if [[ ! -d local/sms_wsj ]]; then
		git clone https://github.com/fgnt/sms_wsj.git local/sms_wsj
	fi
	${python_path} -m pip install -e local/sms_wsj
fi


if [[ $stage -le  1 ]]; then
	echo "Stage 1: Generating SMS_WSJ data"
  . local/prepare_data.sh --wsj0_dir $wsj0_dir --wsj1_dir $wsj1_dir --num_jobs $num_jobs \
   		--sms_wsj_dir $sms_wsj --json_dir $data_dir --python_path $python_path
fi


if [[ $stage -le  2 ]]; then
	if [[ ! -d local/pb_bss ]]; then
		echo "Downloading and installing pb_bss (a model based source separation toolbox)"
		git clone https://github.com/fgnt/pb_bss.git local/pb_bss
		${python_path} -m pip install einops
		${python_path} -m pip install nara_wpe
		${python_path} -m pip install cython
		${python_path} -m pip install -e local/pb_bss[all]
	fi
	mpiexec -n $num_jobs ${python_path} start_evaluation.py --json_path $data_dir/sms_wsj.json
fi