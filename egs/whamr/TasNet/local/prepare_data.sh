#!/bin/bash

wav_dir=tmp
out_dir=tmp
python_path=python

. utils/parse_options.sh

## Download WHAM noises
mkdir -p $out_dir
echo "Download WHAM noises into $out_dir"
# If downloading stalls for more than 20s, relaunch from previous state.
#wget -c --tries=0 --read-timeout=20 https://storage.googleapis.com/whisper-public/wham_noise.zip -P $out_dir
#
#
#echo "Download WHAMR scripts into $out_dir"
#wget https://storage.googleapis.com/whisper-public/whamr_scripts.tar.gz -P $out_dir
#tar -xzvf $out_dir/whamr_scripts.tar.gz -C $out_dir/
#mv $out_dir/whamr_scripts.tar.gz $out_dir/whamr_scripts
#
#wait
#
#echo "Unzip WHAM noises into $out_dir"
#mkdir -p logs
#unzip $out_dir/wham_noise.zip -d $out_dir >> logs/unzip_whamr.log


cd $out_dir/whamr_scripts
echo "Run python scripts to create the WHAM mixtures"
# Requires : Pyloudnorm, Numpy, Scipy, Pandas, Pysoundfile and pyroomacoustics
$python_path -m pip install -r requirements.txt

$python_path create_wham_from_scratch.py \
	--wsj0-root $wav_dir \
	--wham-noise-root $out_dir/wham_noise\
	--output-dir $out_dir
cd -
