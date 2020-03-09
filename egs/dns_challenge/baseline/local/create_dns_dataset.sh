#!/bin/bash

clone_dir=$1
storage_dir=$2

cd $clone_dir/DNS-Challenge
# SED the cfg file to modify windows-like path to linux-like path
sed -i 's+\\+\/+g' noisyspeech_synthesizer.cfg

# Change default saving directories
# We keep the default values for all the rest feel free to modify it.
sed -i 's+./training+'"$storage_dir"'+g' noisyspeech_synthesizer.cfg

# Run the dataset recipe
python -m pip install librosa pandas
python noisyspeech_synthesizer_singleprocess.py