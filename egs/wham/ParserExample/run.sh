#!/bin/bash

python_path=python


# The following lines declare the variables from conf.yml in bash to be parsed from CL.
# With set -a, k=v is equivalent to export k=v. We us that to later get the
# updated values in python with os.environ
set -a
eval `$python_path ./utils/yaml2bash.py local/conf.yml`
set +a

# The three lines above, can be called in a single bash script
# Or we could modify parse_options.sh to accept a YAML file.
# That's not the point, just showing how it works for now.

# Parse all arguments, including the ones from conf.yml
. utils/parse_options.sh

# Get back all the argument in the form --k new_v to pass to training script.
# all args contains all the variables from conf.yml, expected from train.py.
all_args=$($python_path utils/yaml2bash.py local/conf.yml --make_args)
#echo $all_args

$python_path train.py --exp_dir whatever $all_args

