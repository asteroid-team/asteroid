#!/bin/bash

storage_dir=
python_path=python

. ./utils/parse_options.sh

# Clone Libri_VAD repo
git clone https://github.com/asteroid-team/Libri_VAD

# Run generation script
cd Libri_VAD
. run.sh $storage_dir
