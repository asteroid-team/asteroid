#!/bin/bash

# Directory to store LibriSpeech raw files
storage_dir=tmp
# Directory to store Libri2mix
out_dir=tmp


python_path=python

. ../utils/parse_options.sh

## Download WHAM noises
mkdir -p $out_dir
mkdir -p $storage_dir
echo "Download LibiSpeech_train_100_hours_clean dataset into $storage_dir"
# If downloading stalls for more than 20s, relaunch from previous state.
wget -c --tries=0 --read-timeout=20 http://www.openslr.org/resources/12/train-clean-360.tar.gz -P $storage_dir
tar -xzvf $storage_dir/train-clean-100.tar.gz -C $storage_dir

echo "Download LibiSpeech_test_clean dataset into $storage_dir"
# If downloading stalls for more than 20s, relaunch from previous state.
wget -c --tries=0 --read-timeout=20 http://www.openslr.org/resources/12/test-clean.tar.gz -P $storage_dir
tar -xzvf $storage_dir/test-clean.tar.gz -C $storage_dir


echo "Download LibiSpeech_cv_clean dataset into $storage_dir"
# If downloading stalls for more than 20s, relaunch from previous state.
wget -c --tries=0 --read-timeout=20 http://www.openslr.org/resources/12/dev-clean.tar.gz -P $storage_dir
tar -xzvf $storage_dir/dev-clean.tar.gz -C $storage_dir

echo "Download LibiSpeech metadata into $storage_dir"
# If downloading stalls for more than 20s, relaunch from previous state.
wget -c --tries=0 --read-timeout=20 http://www.openslr.org/resources/12/raw-metadata.tar.gz -P $storage_dir
tar -xzvf $storage_dir/raw-metadata.tar.gz -C $storage_dir


echo "Run python scripts to create the Libri2mix sources and mixtures"
# Requires : Numpy, Scipy, Pandas, and Pysoundfile

$python_path create_libri2mix_from_scratch.py --out_dir $out_dir --in_dir $storage_dir\
