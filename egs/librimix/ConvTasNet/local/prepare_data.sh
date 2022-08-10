#!/bin/bash

storage_dir=
n_src=
python_path=python

. ./utils/parse_options.sh

if [[ $n_src -le  1 ]]
then
  changed_n_src=2
else
  changed_n_src=$n_src
fi

$python_path local/create_local_metadata.py --librimix_dir $storage_dir/Libri$changed_n_src"Mix"

$python_path local/get_text.py \
  --libridir $storage_dir/LibriSpeech \
  --split test-clean \
  --outfile data/test_annotations.csv
