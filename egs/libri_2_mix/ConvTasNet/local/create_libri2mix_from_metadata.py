import os
import argparse
import soundfile as sf
import pandas as pd
import numpy as np
from scipy.signal import resample_poly
import random


def main(arguments):
    # Get librispeech root path
    librispeech_root_path = arguments.librispeech_root_path

    # Get dataset root path
    dataset_root_path = arguments.dataset_root_path

    # Get the desired frequencies
    freqs = arguments.freqs

    # Get the desired modes
    modes = arguments.modes

    # Generate source
    generate_mixtures(librispeech_root_path, dataset_root_path, freqs, modes)


def resample_list(sources_list, freq):
    # Create the resampled list
    resampled_list = []

    # Resample each source
    for source in sources_list:
        resampled_list.append(resample_poly(source, freq, 16000))

    return resampled_list


def get_longest_source(sources_list):
    # Get the max length
    len_list = [len(source) for source in sources_list]
    index = len_list.index(max(len_list))

    return index


def get_shortest_source(sources_list):
    # Get the max length
    len_list = [len(source) for source in sources_list]
    index = len_list.index(min(len_list))

    return index


def pad_max(sources_list, index):
    # Copy the list
    sources_list_copy = sources_list.copy()

    # Get target length
    target_length = len(sources_list_copy[index])

    # Initialize result list
    sources_list_max = [sources_list_copy[index]]

    # Delete the element that win't change
    sources_list_copy.pop(index)

    # Change the others and add them to the result list
    for source in sources_list_copy:
        # 0 symmetrical padding
        start = random.randint(0, target_length - len(source))
        sources_list_max.append(
            np.pad(source, (start, target_length - len(source) - start)))
    return sources_list_max


def cut_to_min(sources_list, index):
    # Copy the list and get the target length
    sources_list_copy = sources_list.copy()
    target_length = len(sources_list_copy[index])

    # Initialize result list
    sources_list_min = [sources_list_copy[index]]

    # Delete the element that win't change
    sources_list_copy.pop(index)

    # Change the others and add them to the result list
    for source in sources_list_copy:
        # Start from the beginning cut at target length
        sources_list_min.append(source[:target_length])

    return sources_list_min


def mix(sources_list, mode):
    # Change sources list according to mode
    if mode == 'min':

        index = get_shortest_source(sources_list)

        # Cut to min source length
        sources_list_reshaped = cut_to_min(sources_list, index)

    else:
        index = get_longest_source(sources_list)

        # Pad to max source length
        sources_list_reshaped = pad_max(sources_list, index)

    # Initialize mixture
    mixture = np.zeros_like(sources_list_reshaped[0])

    for source in sources_list_reshaped:
        mixture += source

    return mixture


def generate_mixtures(librispeech_root_path, dataset_root_path, freqs, modes):
    # Get the metadata directory path
    metadata_dir_path = os.path.join(dataset_root_path, 'metadata')

    # Get metadata files name
    metadata_files_names = os.listdir(metadata_dir_path)

    # We will check if the mixtures don't already exist we create a list that
    # will contain the metadata files that already have been used
    already_exist = []

    # Create directories according to parameters
    for freq in freqs:

        # Create frequency directory
        freq_path = os.path.join(dataset_root_path, freq)
        os.makedirs(freq_path, exist_ok=True)

        for mode in modes:
            # Create mode directory
            mode_path = os.path.join(freq_path, mode)
            os.makedirs(mode_path, exist_ok=True)

            for metadata_files_name in metadata_files_names:
                # Create directory name
                directory_name = metadata_files_name.split('.')[0]

                # Make the directory according to the metadata file
                directory_path = os.path.join(mode_path, directory_name)
                try:
                    os.mkdir(directory_path)
                except FileExistsError:
                    already_exist.append(metadata_files_name)
                    print(f"The mixtures from the {metadata_files_name} file"
                          f" already exist, the metadafile will be ignored")

    for element in already_exist:
        metadata_files_names.remove(element)

    for metadata_files_name in metadata_files_names:

        # Get metadata file path
        metadata_file_path = os.path.join(metadata_dir_path,
                                          metadata_files_name)

        # Read the csv file
        metadata_file = pd.read_csv(metadata_file_path)

        # Go throw the metadata file and generate mixtures
        for index, row in metadata_file.iterrows():

            # Get info about the mixture
            Mixture_ID = row['Mixture_ID']
            sources_paths = row['Path_list']
            sources_paths = sources_paths.strip("['")
            sources_paths = sources_paths.strip("']")
            sources_paths = sources_paths.replace("'", "")
            sources_paths = sources_paths.split(', ')
            sources_list = []

            # Read the files to make the mixture
            for sources_path in sources_paths:
                sources_path = os.path.join(librispeech_root_path,
                                            sources_path)
                source, _ = sf.read(sources_path, dtype='float32', )
                sources_list.append(source)

            for freq in freqs:

                # Get the frequency directory path
                freq_path = os.path.join(dataset_root_path, freq)

                # Transform freq = "16K" into 16000
                freq = int(freq.strip('K')) * 1000

                # Resample the sources
                sources_list = resample_list(sources_list, freq)

                # Mix
                for mode in modes:
                    mode_path = os.path.join(freq_path, mode)
                    mixtures = mix(sources_list, mode)

                    directory_name = metadata_files_name.split('.')[0]
                    directory_path = os.path.join(mode_path, directory_name)

                    # Save the mixture
                    mixture_path = os.path.join(directory_path, Mixture_ID +
                                                '.flac')
                    sf.write(mixture_path, mixtures, freq)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--librispeech_root_path', type=str, default=None,
                        help='Path to librispeech root directory')
    parser.add_argument('--dataset_root_path', type=str, default=None,
                        help='Path to the desired dataset root directory')
    parser.add_argument('--freqs', nargs='+', default=['8K'],
                        help='--freqs 16K 8K will create 2 directories 8K '
                             'and 16K')
    parser.add_argument('--modes', nargs='+', default=['min'],
                        help='--modes min max will create 2 directories in '
                             'each freq directory')
    args = parser.parse_args()
    main(args)
