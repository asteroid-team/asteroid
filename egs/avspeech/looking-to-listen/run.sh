#!/bin/bash

# Exit on error
set -e
set -o pipefail

parse_boolean() {
	# parses yaml boolean to flag.
	# e.g. audio_set=True to --use-audio-set for argparse

        var=$1
        flag=$2
        var=$(echo $var | awk '{print tolower($0)}')
        var=$([ "$var" == "true" ] && echo "$flag" || echo "")
        echo $var
}

get_attribute() {
	# get an attribute from local/conf.yml

	pushd $root_dir > /dev/null
	attribute=$1
	value=$(grep  $attribute local/data_prep.yml | awk  '{print $2}' | sed 's/"//g')
	popd > /dev/null
	echo $value
}

check_dependency() {
	# check if dependeny exists for
	# ffmpeg, sox, youtube-dl
	cmd=$1

	type $cmd  >/dev/null 2>&1 || { echo >&2 "$cmd is required but it's not installed. Aborting."; exit 1; }
}

root_dir=$(pwd)

python_path=python
loader_dir=$root_dir/local/loader
stage=0
gpu_ids=0
exp_dir=exp/logdir
tag=
install_flag=false
storage_dir=${STORAGE_DIR:-"storage_dir"}

. utils/parse_options.sh

if [ "$install_flag" = true ]; then
	# install python dependencies
	echo "installing python dependencies"
	$python_path -m pip install -r local/requirements.txt > /dev/null
fi

if [[ $stage -le 0 ]]; then

	# Setup the structure for data downloading, pre-processing
	mkdir -p $storage_dir/storage/{video,audio,embed,spec,mixed} data/{audio_set,audio_visual} exp logs

	echo "Stage 0: Setting up structure and downloading files."
	pushd $loader_dir > /dev/null

	n_jobs=$(get_attribute "download_jobs")
	path=$(get_attribute "download_path")
	download_start=$(get_attribute "download_start")
	download_end=$(get_attribute "download_end")

	echo "Downloading...(Interruptible)"
	check_dependency ffmpeg
	check_dependency youtube-dl
	python3 download.py --jobs $n_jobs --path $path \
			    --start $download_start \
			    --end $download_end

	cd $root_dir
	vid_dir=$storage_dir/storage/video
	n_files=$(ls -1q $vid_dir/*_final.mp4 | wc -l)
	echo "Total files: $n_files"
fi


if [[ $stage -le 1 ]]; then
	echo "Stage 1: Extracting audio and Mixing audio files."
	cd $loader_dir

	n_jobs=$(get_attribute "extract_jobs")
	sampling_rate=$(get_attribute "extract_sampling_rate")
	input_audio_channel=$(get_attribute "extract_input_audio_channel")
	audio_extension=$(get_attribute "extract_audio_extension")
	duration=$(get_attribute "extract_duration")

	echo "Extracting audio..."
	check_dependency ffmpeg
	python3 extract_audio.py --jobs $n_jobs \
				 --sampling-rate $sampling_rate \
				 --audio-channel $input_audio_channel \
				 --audio-extension $audio_extension --duration $duration

	remove_random_chance=$(get_attribute "mix_remove_random_chance")
	use_audio_set=$(get_attribute "mix_use_audio_set")
	file_limit=$(get_attribute "mix_file_limit")
	validation_size=$(get_attribute "mix_validation_size")

	use_audio_set=$(parse_boolean $use_audio_set '--use-audio-set')

	echo "Mixing audio...(Interruptible)"
	[ "$use_audio_set" = "--use-audio-set" ] && check_dependency sox
	python3 audio_mixer_generator.py --remove-random $remove_random_chance \
					 --file-limit $file_limit \
					 --validation-size $validation_size $use_audio_set

	echo "Cleaning up audio..."
	python3 remove_empty_audio.py
	cd $root_dir
fi

if [[ $stage -le 2 ]]; then
	echo "Stage 2: Preprocessing video. Extracting faces."
	cd $loader_dir

	use_cuda=$(get_attribute "face_cuda")
	use_half=$(get_attribute "face_use_half")
	corrupt_file_path=$(get_attribute "face_corrupt_file_path")

	use_cuda=$(parse_boolean $use_cuda '--cuda')
	use_half=$(parse_boolean $use_half '--use-half')

	python3 generate_video_embedding.py --corrupt-file $corrupt_file_path $use_cuda $use_half

	python3 remove_corrupt.py
	cd $root_dir
fi

n_src=$(get_attribute "n_src")
if [[ -z ${tag} ]]; then
	# Generate a random ID for the run if no tag is specified
	uuid=$($python_path -c 'import uuid; print(str(uuid.uuid4())[:8])')
	clean_dir_name=$(echo $storage_dir | sed 's/\//_/g')
	tag=${n_src}_${clean_dir_name}_${uuid}
fi
exp_dir="${exp_dir}_${tag}"

if [[ $stage -le 3 ]]; then
	echo "Stage 3: Training"

	mkdir -p $exp_dir
	python3 train.py --gpus $gpu_ids --exp_dir $exp_dir \
			 --n-src $n_src | tee logs/train_${tag}.log
fi

if [[ $stage -le 4 ]]; then
	echo "Stage 4: Validation"

	mkdir -p $exp_dir
	python3 eval.py --gpus $gpu_ids --exp_dir $exp_dir \
        		--n-src $n_src | tee logs/val_${tag}.log
fi

