#!/bin/bash

root_dir=$(PWD)
loader_dir=$root_dir/local/loader
stage=0
gpu_ids=0
exp_dir=data/logdir

parse_boolean() {
	var=$1
	flag=$2
	var=$(awk -vvar=$var '{print tolower(var)}')
	var=$([ "$var" == "true" ] && echo "$flag" || echo "")
}

get_attribute() {
	pushd $root_dir
	attribute=$1
	value=$(grep  $attribute local/conf.yml | awk  '{print $2}')
	popd
	echo $value
}

if [[ $stage -le  0 ]]; then
	echo "Stage 0: Setting up structure and downloading files."
	cd $loader_dir
	# Setup the structure for data downloading, pre-processing
	mkdir -p ../../data/{train,val}/{audio,embed,spec,mixed} ../../data/{audio_set,audio_visual}

	n_jobs=$(get_attribute "download_jobs")
	path=$(get_attribute "download_path")
	vid_dir=$(get_attribute "download_vid_dir")
	download_start=$(get_attribute "download_start")
	download_end=$(get_attribute "download_end")

	echo "Downloading...(Interruptible)"
	python3 download.py --jobs $n_jobs --path $path \
			    --vid-dir $vid_dir --start $download_start \
			    --end $download_end

	n_files=$(ls $vid_dir -1q *_final.mp4 | wc -l)
	echo "Total files: $n_files"
	cd $root_dir
fi


if [[ $stage -le  1 ]]; then
	echo "Stage 1: Extracting audio and Mixing audio files."
	cd $loader_dir

	n_jobs=$(get_attribute "extract_jobs")
	vid_dir=$(get_attribute "download_vid_dir") # extract from download path
	audio_dir=$(get_attribute "extract_audio_dir")
	sampling_rate=$(get_attribute "extract_sampling_rate")
	input_audio_channel=$(get_attribute "extract_input_audio_channel")
	audio_extension=$(get_attribute "extract_audio_extension")
	duration=$(get_attribute "extract_duration")

	echo "Extracting audio...(Interruptible)"
	python3 extract_audio.py --jobs $n_jobs --aud-dir $audio_dir \
				 --vid-dir $vid_dir --sampling-rate $sampling_rate \
				 --audio-channel $input_audio_channel \
				 --audio-extension $audio_extension --duration $duration

	remove_random_chance=$(get_attribute "mix_remove_random_chance")
	use_audio_set=$(get_attribute "mix_use_audio_set")
	file_limit=$(get_attribute "mix_file_limit")
	validation_size=$(get_attribute "mix_validation_size")

	echo "Mixing audio...(Interruptible)"
	python3 audio_mixer_generator.py --remove-random $remove_random_chance \
					 --use-audio-set $use_audio_set \
					 --file-limit $file_limit \
					 --validation-size $validation_size

	echo "Cleaning up audio..."
	python3 remove_empty_audio.py
	cd $root_dir
fi

if [[ $stage -le 2 ]]; then
	echo "Stage 2: Preprocessing video. Extracting faces."
	cd $loader_dir

	vid_dir=$(get_attribute "download_vid_dir") # extract from download path
	use_cuda=$(get_attribute "face_cuda")
	use_half=$(get_attribute "face_use_half")
	corrupt_file_path=$(get_attribute "face_corrupt_file_path")

	use_cuda=$(parse_bool $use_cuda '--cuda')
	use_half=$(parse_bool $use_half '--use-half')

	python3 generate_video_embedding.py --video-dir $vid_dir \
					    --corrupt-file $corrupt_file_path $use_cuda $use_half
	cd $root_dir
fi

if [[ $stage -le 3 ]]; then
	echo "Stage 3: Training"

	python3 train.py --gpus $gpu_ids --exp_dir $exp_dir
fi

if [[ $stage -le 4 ]]; then
	echo "Stage 4: Validation"

	python3 eval.py --gpus $gpu_ids --exp_dir $exp_dir
fi
