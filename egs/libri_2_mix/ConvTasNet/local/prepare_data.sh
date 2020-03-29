#!/bin/bash

. ../utils/parse_options.sh

python_path=python

# shellcheck disable=SC1007

# Create a directory and do all the download there
storage_dir= storage_dir/libri
mkdir -p $storage_dir/libri

echo "Download LibiSpeech_train_100_hours_clean dataset into $storage_dir"
# If downloading stalls for more than 20s, relaunch from previous state.
wget -c --tries=0 --read-timeout=20 http://www.openslr.org/resources/12/train-clean-100.tar.gz -P $storage_dir
tar -xzvf $storage_dir/train-clean-100.tar.gz -C $storage_dir
rm -rf $storage_dir/train-clean-100.tar.gz

echo "Download LibiSpeech_test_clean dataset into $storage_dir"
# If downloading stalls for more than 20s, relaunch from previous state.
wget -c --tries=0 --read-timeout=20 http://www.openslr.org/resources/12/test-clean.tar.gz -P $storage_dir
tar -xzvf $storage_dir/test-clean.tar.gz -C $storage_dir
rm -rf  $storage_dir/test-clean.tar.gz

echo "Download LibiSpeech_cv_clean dataset into $storage_dir"
# If downloading stalls for more than 20s, relaunch from previous state.
wget -c --tries=0 --read-timeout=20 http://www.openslr.org/resources/12/dev-clean.tar.gz -P $storage_dir
tar -xzvf $storage_dir/dev-clean.tar.gz -C $storage_dir
rm -rf $storage_dir/dev-clean.tar.gz
