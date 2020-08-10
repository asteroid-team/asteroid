#!/bin/bash

# Exit on error
set -e
set -o pipefail


storage_dir= # Main storage directory (Need disk space)
wsj_dir= # WSJ DIR (if wsj0 and wsj1 have been merged)

# Following wsj_dirs have to be defined, if wsj0 and wsj1 have not been merged
wsj0_dir=$wsj_dir
wsj1_dir=$wsj_dir

num_jobs=$(nproc --all)

stage=0

python_path=python
. utils/parse_options.sh || exit 1;

data_dir=data  # Local data directory (Not much disk space required)
sms_wsj=${storage_dir}/DATA/sms_wsj/ # Directory where to save SMS-WSJ wav files

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1


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
