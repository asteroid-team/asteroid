#!/bin/bash

wav_dir=tmp

. utils/parse_options.sh

## Download WHAM noises
mkdir -p $wav_dir
echo "Download WHAM noises into $wav_dir"
#wget https://storage.googleapis.com/whisper-public/wham_noise.zip -P $wav_dir >> $wav_dir/noise_wget.log &

echo "Download WHAM scripts into $wav_dir"
wget https://storage.googleapis.com/whisper-public/wham_scripts.tar.gz -P $wav_dir
mkdir -p $wav_dir/wham_scripts && tar -xzvf $wav_dir/wham_scripts.tar.gz -C $wav_dir/wham_scripts
mv $wav_dir/wham_scripts.tar.gz $wav_dir/wham_scripts

wait

#unzip $wav_dir/wham_noise.zip && mv $wav_dir/wham_noise.zip $wav_dir/wham_noise

#run scripts
## Preprocessing (do all the cases)
## Find out wsj-3mix as well no?
#