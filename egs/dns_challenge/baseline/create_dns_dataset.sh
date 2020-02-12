#!/bin/bash

storage_dir=$1

# SED the cfg file to modify windows-like path to linux-like path
sed -i 's+\\+\/+g' noisyspeech_synthesizer.cfg

# Change default saving directories
# We keep the default values for all the rest feel free to modify it.
sed -i 's+./training+'"$storage_dir"'+g'

# Run the dataset recipe. (which python??)
python noisyspeech_synthesizer_singleprocess.py