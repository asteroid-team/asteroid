#!/bin/bash


storage_dir=
# If you want to clone the DNS-Challenge repo somewhere different
clone_dir=  # optional

stage=0


. ./utils/parse_options.sh

recipe_dir=$PWD

if [[ -z ${clone_dir} ]]; then
	clone_dir=$storage_dir
fi

if [[ $stage -le  0 ]]; then
  echo "Stage 0 : Install git-lfs"
  . ./local/install_git_lfs.sh
fi


if [[ $stage -le  1 ]]; then
  echo "Stage 1 : Download the data (this will take a while)"
  . ./local/download_data.sh $clone_dir
fi

if [[ $stage -le  2 ]]; then
  echo "Stage 2 : Create the dataset"
  cd $clone_dir/DNS-Challenge
  . ./local/create_dns_dataset.sh
  cd $recipe_dir
fi

dumpdir=data

if [[ $stage -le  3 ]]; then
  echo "Stage 3 : preprocess the dataset"
  python local/preprocess_dns.py --data_dir $storage_dir --json_dir $dumpdir
fi

if [[ $stage -le  4 ]]; then
  echo "Stage 4 : Train"

fi

if [[ $stage -le  5 ]]; then
  echo "Stage 5 : Evaluate"
fi