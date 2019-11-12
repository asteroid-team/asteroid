#!/bin/bash

wav_dir=tmp
out_dir=tmp

. ../utils/parse_options.sh

## Download WHAM noises
mkdir -p $wav_dir
echo "Download WHAM noises into $wav_dir"
wget https://storage.googleapis.com/whisper-public/wham_noise.zip -P $wav_dir

echo "Download WHAM scripts into $wav_dir"
wget https://storage.googleapis.com/whisper-public/wham_scripts.tar.gz -P $wav_dir
mkdir -p $wav_dir/wham_scripts && tar -xzvf $wav_dir/wham_scripts.tar.gz -C $wav_dir/wham_scripts
mv $wav_dir/wham_scripts.tar.gz $wav_dir/wham_scripts

wait

unzip $wav_dir/wham_noise.zip && mv $wav_dir/wham_noise.zip $wav_dir/wham_noise

echo "Run python scripts to create the WHAM mixtures"
# Requires : Numpy, Scipy, Pandas, and Pysoundfile
python $wav_dir/wham_scripts/create_wham_from_scratch.py \
	--wsj0-root $wav_dir \
	--wham-noise-root $wav_dir/wham_noise\
	--output-dir $out_dir
