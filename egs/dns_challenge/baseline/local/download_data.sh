#!/bin/bash

clone_dir=$1

recipe_dir=$PWD
cd $clone_dir

# Clone repo
git clone https://github.com/microsoft/DNS-Challenge
cd DNS-Challenge

# Run lfs stuff in the repo
git lfs install
git lfs track "*.wav"
git add .gitattributes

# Go back to the recipe
cd $recipe_dir