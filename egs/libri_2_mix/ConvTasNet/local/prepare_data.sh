#!/bin/bash

storage_dir=

. ./utils/parse_options.sh

# Create a directory and do all the download there
mkdir -p $storage_dir

echo "Download LibiSpeech/train-clean-100 into $storage_dir"
# If downloading stalls for more than 20s, relaunch from previous state.
wget -c --tries=0 --read-timeout=20 http://www.openslr.org/resources/12/train-clean-100.tar.gz -P $storage_dir
tar -xzf $storage_dir/train-clean-100.tar.gz -C $storage_dir
rm -rf $storage_dir/train-clean-100.tar.gz

echo "Download LibiSpeech/test-clean into $storage_dir"
# If downloading stalls for more than 20s, relaunch from previous state.
wget -c --tries=0 --read-timeout=20 http://www.openslr.org/resources/12/test-clean.tar.gz -P $storage_dir
tar -xzf $storage_dir/test-clean.tar.gz -C $storage_dir
rm -rf $storage_dir/test-clean.tar.gz

echo "Download LibiSpeech/dev-clean into $storage_dir"
# If downloading stalls for more than 20s, relaunch from previous state.
wget -c --tries=0 --read-timeout=20 http://www.openslr.org/resources/12/dev-clean.tar.gz -P $storage_dir
tar -xzf $storage_dir/dev-clean.tar.gz -C $storage_dir
rm -rf $storage_dir/dev-clean.tar.gz

echo "Download LibiSpeech_cv_clean dataset into storage_dir"
# If downloading stalls for more than 20s, relaunch from previous state.
wget -c --tries=0 --read-timeout=20 http://www.openslr.org/resources/12/train-clean-360.tar.gz -P $storage_dir
tar -xzvf $storage_dir/train-clean-360.tar.gz -C $storage_dir
rm -rf $storage_dir/train-clean-360.tar.gz
