#!/bin/bash
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file is an adaptation of the official dataset preparation FUSS scripts in
# https://github.com/google-research/sound-separation/tree/master/datasets/fuss


# Set the root directory to store the dataset
ROOT_DIR=$1
# When working with development data, the dataset you want to work with is set to true
# both variables are set to true if you decide to download both datasets
dry_mixtures=$2
reverberated_mixtures=$3


# No need to change anything below this line
. utils/parse_options.sh

# Download directory to download data into.
DOWNLOAD_DIR=${ROOT_DIR}/download


# This is the main directory where the fixed dev data will reside.
DEV_DATA_DIR=${ROOT_DIR}/fuss_dev
# The ssdata archive file is assumed to include a top folder named ssdata.

# Development dataset flag check
DEV_FLAG=0

SSDATA_URL="https://zenodo.org/record/3743844/files/FUSS_ssdata.tar.gz"

# The archive file is assumed to include a top folder named ssdata_reverb.
SSDATA_REVERB_URL="https://zenodo.org/record/3743844/files/FUSS_ssdata_reverb.tar.gz"

SS_DIR=${DEV_DATA_DIR}/ssdata
SS_REVERB_DIR=${DEV_DATA_DIR}/ssdata_reverb

mkdir -p ${DOWNLOAD_DIR}
mkdir -p ${DEV_DATA_DIR}

echo 'Getting development data'
if [ "$dry_mixtures" = true ]; then
    echo 'Dry mixtures will be downloaded'
    # Download and unarchive dry source data for development
    if [ ! -s ${DOWNLOAD_DIR}/ssdata.tar.gz ]; then
        wget -O ${DOWNLOAD_DIR}/ssdata.tar.gz ${SSDATA_URL}
    else
        echo "${DOWNLOAD_DIR}/ssdata.tar.gz exists, skipping download."
    fi

    if [ ! -d ${SS_DIR} ]; then
        tar xzf ${DOWNLOAD_DIR}/ssdata.tar.gz -C ${DEV_DATA_DIR}
    else
        echo "${SS_DIR} directory exists, skipping unarchiving."
    fi
    DEV_FLAG=1
fi

if [ "$reverberated_mixtures" = true ]; then
    echo 'Reverberated mixtures will be downloaded'
    # Download and unarchive reverberated source data for development
    if [ ! -s ${DOWNLOAD_DIR}/ssdata_reverb.tar.gz ]; then
        wget -O ${DOWNLOAD_DIR}/ssdata_reverb.tar.gz ${SSDATA_REVERB_URL}
    else
        echo "${DOWNLOAD_DIR}/ssdata_reverb.tar.gz exists, skipping download."
    fi

    if [ ! -d ${SS_REVERB_DIR} ]; then
    tar xzf ${DOWNLOAD_DIR}/ssdata_reverb.tar.gz -C ${DEV_DATA_DIR}
    else
        echo "${SS_REVERB_DIR} directory exists, skipping unarchiving."
    fi
    DEV_FLAG=1
fi

if [ $DEV_FLAG -le 0 ]; then
    echo 'Set a value: true or false, to fuss_dry and fuss_reverb in run.sh'
    exit 1
fi