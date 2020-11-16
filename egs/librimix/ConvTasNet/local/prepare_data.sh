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

cd $current_dir
$python_path local/create_local_metadata.py --librimix_dir $storage_dir/Libri$n_src"Mix"


# TODO: check folders and outdir
#dev-clean  test-clean  train-clean-100  train-clean-360
for split in train-360 train-100 dev-clean test-clean; do
  $python_path local/get_text.py --libridir $storage_dir/LibriSpeech --split $split --outdir data/$split
done
