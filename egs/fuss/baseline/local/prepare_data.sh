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


# Set the root directory to store the dataset
ROOT_DIR=data
python_path=python

# Choose to work with development data or to generate augmented data
# Set only one variable to true, the other one must be set to false
GET_DEV_DATA=true
GENERATE_AUGMENTED_DATA=false

# If working with development data, set to true which data you want to download
# set both variables to true if you want to download both datasets
dry_mixtures=false
reverberated_mixtures=false

# If generating augmented data, set the following variables

# Random seed to use for augmented data generation.
RANDOM_SEED=2020

# Number of train, validation adn evaluation examples to generate for data augmentation.
NUM_TRAIN=20000
NUM_VAL=1000
NUM_EVAL_MIX=1000


# No need to change anything below this line
. utils/parse_options.sh

# Download directory to download data into.
DOWNLOAD_DIR=${ROOT_DIR}/download

if [ "$GET_DEV_DATA" = true ]; then
    # This is the main directory where the fixed dev data will reside.
    DEV_DATA_DIR=${ROOT_DIR}/fuss_dev
    # The ssdata archive file is assumed to include a top folder named ssdata.
    
    SSDATA_URL="https://zenodo.org/record/3710392/files/FUSS_ssdata.tar.gz"

    # The archive file is assumed to include a top folder named ssdata_reverb.
    SSDATA_REVERB_URL="https://zenodo.org/record/3710392/files/FUSS_ssdata_reverb.tar.gz"

    SS_DIR=${DEV_DATA_DIR}/ssdata
    SS_REVERB_DIR=${DEV_DATA_DIR}/ssdata_reverb

    mkdir -p ${DOWNLOAD_DIR}
    mkdir -p ${DEV_DATA_DIR}

    echo 'Getting development data'
    if [ "$dry_mixtures" = true ] && [ "$reverberated_mixtures" = true ]; then
        echo 'Dry and reverberated mixtures will be downloaded'

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

    elif [ "$dry_mixtures" = true ]; then
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

    elif [ "$reverberated_mixtures" = true ]; then
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

    else
        echo 'Set a value to dry_mixtures and reverberated_mixtures'
        exit 1
    fi

elif [ "$GENERATE_AUGMENTED_DATA" = true ]; then
    echo 'Generating augmented data'
    # This is the main directory where the single source files and room impulse
    # responses that will be used in data augmentation will be downloaded.
    RAW_DATA_DIR=${ROOT_DIR}/fuss_sources_and_rirs

    # This is the main directory where the augmented data will reside.
    # _${RANDOM_SEED} will be appended to this path, so multiple versions
    # of augmented data can be generated by only changing the random seed.
    AUG_DATA_DIR=${ROOT_DIR}/fuss_augment

    # The fsd data archive file is assumed to include a top folder named fsd_data.
    FSD_DATA_URL="https://zenodo.org/record/3710392/files/FUSS_fsd_data.tar.gz"

    # The rir data archive file is assumed to include a top folder named rir_data.
    RIR_DATA_URL="https://zenodo.org/record/3710392/files/FUSS_rir_data.tar.gz"

    # Get raw data
    FSD_DIR=${RAW_DATA_DIR}/fsd_data
    RIR_DIR=${RAW_DATA_DIR}/rir_data

    mkdir -p ${DOWNLOAD_DIR}
    mkdir -p ${RAW_DATA_DIR}

    # Download and unarchive FSD data.
    if [ ! -s ${DOWNLOAD_DIR}/fsd_data.tar.gz ]; then
        wget -O ${DOWNLOAD_DIR}/fsd_data.tar.gz ${FSD_DATA_URL}
    else
        echo "${DOWNLOAD_DIR}/fsd_data.tar.gz exists, skipping download."
    fi

    if [ ! -d ${FSD_DIR} ]; then
        tar xzf ${DOWNLOAD_DIR}/fsd_data.tar.gz -C ${RAW_DATA_DIR}
    else
        echo "${FSD_DIR} directory exists, skipping unarchiving."
    fi

    # Download and unarchive RIR data.
    if [ ! -s ${DOWNLOAD_DIR}/rir_data.tar.gz ]; then
        wget -O ${DOWNLOAD_DIR}/rir_data.tar.gz ${RIR_DATA_URL}
    else
        echo "${DOWNLOAD_DIR}/rir_data.tar.gz exists, skipping download."
    fi

    if [ ! -d ${RIR_DIR} ]; then
        tar xzf ${DOWNLOAD_DIR}/rir_data.tar.gz -C ${RAW_DATA_DIR}
    else
        echo "${RIR_DIR} directory exists, skipping unarchiving."
    fi

    wait

    # Download FUSS Dataset scripts to generate the dataset
    FUSS_SCRIPTS_DIR=${ROOT_DIR}/fuss_scripts
    mkdir -p ${FUSS_SCRIPTS_DIR}

    echo "Download FUSS scripts into ${FUSS_SCRIPTS_DIR}"
    wget -O ${FUSS_SCRIPTS_DIR}/fuss_scripts.zip https://github.com/google-research/sound-separation/archive/master.zip -P ${FUSS_SCRIPTS_DIR}
    unzip ${FUSS_SCRIPTS_DIR}/fuss_scripts.zip 'sound-separation-master/datasets/fuss/*.py' -d ${FUSS_SCRIPTS_DIR}
    mv ${FUSS_SCRIPTS_DIR}/sound-separation-master/datasets/fuss/* ${FUSS_SCRIPTS_DIR}
    rm -r ${FUSS_SCRIPTS_DIR}/sound-separation-master

    echo "Run python sripts to create FUSS reberved mixtures"
    # Install dependencies
    # Requires: Numpy, Scipy, Pandas, Soundfile, Scaper

    # run scaper

    # Actual GEN_DATA_DIR to use for data generation outputs.
    GEN_DATA_DIR=${AUG_DATA_DIR}_${RANDOM_SEED}
    MIX_DIR=${GEN_DATA_DIR}/ssdata
    REVERB_MIX_DIR=${GEN_DATA_DIR}/ssdata_reverb

    # Makes foreground and background wav file lists from fsd_data used as
    # foreground and background events in scaper.
    $python_path ${FUSS_SCRIPTS_DIR}/make_fg_bg_file_lists.py --data_dir ${FSD_DIR}

    # Runs scaper to obtain the desired amount of example mixed signals.
    # It also saves the individual source wavs used in mixture wavs.
    $python_path ${FUSS_SCRIPTS_DIR}/make_ss_examples.py -f ${FSD_DIR} -b ${FSD_DIR} \
    -o ${MIX_DIR} --allow_same 1 --num_train ${NUM_TRAIN} \
    --num_validation ${NUM_VAL} --num_eval ${NUM_EVAL_MIX} \
    --random_seed ${RANDOM_SEED}

    # Reverberate and remix data.

    # Define variables for this part.
    MIX_INFO=${REVERB_MIX_DIR}/mix_info.txt
    SRC_LIST=${REVERB_MIX_DIR}/src_list.txt
    RIR_LIST=${REVERB_MIX_DIR}/rir_list.txt
    LOG_DIR=${REVERB_MIX_DIR}/log

    if [ ! -d ${REVERB_MIX_DIR} ]; then
        mkdir -p ${REVERB_MIX_DIR}
    else
        echo "${REVERB_MIX_DIR} exists. Please delete and re-run. Exiting."
        exit 1
    fi

    # Form src_list from train_example_list.txt validation_example_list.txt
    # and eval_example_list.txt produced by make_ss_examples.py.

    cp ${MIX_DIR}/*_example_list.txt ${REVERB_MIX_DIR}
    cat ${REVERB_MIX_DIR}/*_example_list.txt > ${SRC_LIST}

    mkdir -p ${LOG_DIR}

    echo "First make a mix_info file ${MIX_INFO}. Inspect the file to be sure."

    $python_path ${FUSS_SCRIPTS_DIR}/reverberate_and_mix.py -s ${MIX_DIR} -r ${RIR_DIR} \
    -o ${REVERB_MIX_DIR} --write_mix_info ${MIX_INFO} --write_rirs ${RIR_LIST} \
    --read_sources ${SRC_LIST} --random_seed ${RANDOM_SEED}

    echo "Running generation with ${NPARTS_REVERB} in the background"
    echo "with produced mix_info file ${MIX_INFO}"

    for part in $(seq 0 $(( NPARTS_REVERB - 1 ))); do
        $python ${FUSS_SCRIPTS_DIR}/reverberate_and_mix.py -s ${MIX_DIR} -r ${RIR_DIR} \
        -o ${REVERB_MIX_DIR} --read_mix_info ${MIX_INFO} \
        --part ${part} --nparts ${NPARTS_REVERB} --chat 1 > \
        ${LOG_DIR}/rev_and_mix_out_${part}_of_${NPARTS_REVERB} 2>&1 &
    done

    echo "Waiting for ${NPARTS_REVERB} background processes to finish! Check logs"
    echo "in ${LOG_DIR}."
    
    wait

    echo "Mixture generation has been done!"

    # Let's check the reverb_mix folder by check_and_fix_folder.py
    # There may be small mixture consistency errors due to writing sources in int16
    # files whereas the mixture is calculated using floats originally.
    $python_path ${FUSS_SCRIPTS_DIR}/check_and_fix_folder.py \
    -sd ${REVERB_MIX_DIR} -sl ${SRC_LIST}

else
    echo 'Set a value to GET_DEV_DATA and GENERATE_AUGMENTED_DATA'
    exit 1
fi

