#!/bin/bash
dampvsep_root=
python_path=python

. ./utils/parse_options.sh
current_dir=$(pwd)
# Clone preprocessed DAMP-VSEP-Singles repo
git clone https://github.com/groadabike/DAMP-VSEP-Singles.git

# Generate the splits
. DAMP-VSEP-Singles/generate_dampvsep_singles.sh $dampvsep_root metadata $python_path