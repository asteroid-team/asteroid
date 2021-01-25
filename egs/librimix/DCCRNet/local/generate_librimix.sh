#!/bin/bash

storage_dir=
n_src=
python_path=python

. ./utils/parse_options.sh

current_dir=$(pwd)
# Clone LibriMix repo
git clone https://github.com/JorisCos/LibriMix

# Run generation script
cd LibriMix
. generate_librimix.sh $storage_dir
