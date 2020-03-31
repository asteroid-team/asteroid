import os
import argparse
import soundfile as sf
import pandas as pd
import numpy as np
from scipy.signal import resample_poly
import pyloudnorm as pyln

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


def main(arguments):
    # Get librispeech root path
    librispeech_root_path = arguments.librispeech_root_path
    librispeech_root_path = 'D:/LibriSpeech'
    # Get dataset root path
    dataset_root_path = arguments.dataset_root_path
    dataset_root_path = 'D:/libri2mix'
    # Get the desired frequencies
    freqs = arguments.freqs
    freqs = ['16K', '8K']
    # Get the desired modes
    modes = arguments.modes
    modes = ['min', 'max']
    # Generate source
    generate_mixtures_and_sources(librispeech_root_path, dataset_root_path,
                                  freqs, modes)


def generate_mixtures_and_sources(
        librispeech_root_path, dataset_root_path, freqs, modes):
    """ Generate mixtures and saves them in dataset_root_path"""
    # Get the metadata directory path
    metadata_dir_path = os.path.join(dataset_root_path, 'metadata')

    # Check the metadata files already used and create directories
    # to store sources and mixtures
    metadata_files_names = create_directories(
        dataset_root_path, metadata_dir_path, freqs, modes)

    for metadata_files_name in metadata_files_names:

        print(f"Creating mixtures from {metadata_files_name}")

        # Get metadata file path
        metadata_file_path = os.path.join(metadata_dir_path,
                                          metadata_files_name)

        # Read the csv file
        metadata_file = pd.read_csv(metadata_file_path)

        # Go through the metadata file and generate mixtures
        for index, row in metadata_file.iterrows():

            # Get info about the mixture
            mixture_id = row['Mixture_ID']
            sources_paths = get_list_from_csv(row['Path_list'], 'str')
            target_loudness_list = get_list_from_csv(
                row['target_loudness_list'], 'float')
            original_loudness_list = get_list_from_csv(
                row['original_loudness_list'], 'float')

            sources_list = []

            # Read the files to make the mixture
            for sources_path in sources_paths:
                sources_path = os.path.join(librispeech_root_path,
                                            sources_path)
                source, _ = sf.read(sources_path, dtype='float32')
                sources_list.append(source)

            for freq in freqs:

                # Get the frequency directory path
                freq_path = os.path.join(dataset_root_path, freq)

                # Transform freq = "16K" into 16000
                freq = int(freq.strip('K')) * 1000

                # Mix
                for mode in modes:
                    # Normalize sources
                    sources_list_norm = loudness_normalize(
                        sources_list, original_loudness_list,
                        target_loudness_list)
                    # Resample the sources
                    sources_list_resampled = resample_list(sources_list_norm,
                                                           freq)
                    # Reshape sources
                    reshaped_sources = fit_lengths(sources_list_resampled,
                                                   mode)
                    # Do the mixture
                    mixture = mix(reshaped_sources)
                    # Path to the mode directory
                    mode_path = os.path.join(freq_path, mode)
                    # Path to the sources directory
                    sources_path = os.path.join(mode_path, 'sources')
                    # Path to the mixtures directory
                    mixtures_path = os.path.join(mode_path, 'mixtures')
                    # Get the directory path where the mixture will be saved
                    directory_name = metadata_files_name.split('.')[0]
                    directory_mixture_path = os.path.join(mixtures_path,
                                                          directory_name)
                    directory_source_path = os.path.join(sources_path,
                                                         directory_name)

                    for i, source in enumerate(reshaped_sources):
                        source_id = mixture_id.split('_')[i]
                        source_path = os.path.join(directory_source_path,
                                                   source_id + '.wav')
                        sf.write(source_path, source, freq)
                    # Save the mixture
                    mixture_path = os.path.join(directory_mixture_path,
                                                mixture_id + '.wav')
                    sf.write(mixture_path, mixture, freq)


def create_directories(dataset_root_path, metadata_dir_path, freqs, modes):
    # Get metadata files name
    metadata_files_names = os.listdir(metadata_dir_path)

    # We will check if the mixtures and sources  don't already exist we create
    # a list that will contain the metadata files that already have been used.
    # You can also specify metadata files to ignore.
    already_exist = []

    # Subdirectories
    subdirs = ['sources', 'mixtures']

    # Create directories according to parameters
    for freq in freqs:

        # Create frequency directory
        freq_path = os.path.join(dataset_root_path, freq)
        os.makedirs(freq_path, exist_ok=True)

        for mode in modes:
            # Create mode directory
            mode_path = os.path.join(freq_path, mode)
            os.makedirs(mode_path, exist_ok=True)
            for subdir in subdirs:
                # Create mixtures and sources directories
                subdir_path = os.path.join(mode_path, subdir)
                os.makedirs(subdir_path, exist_ok=True)
                for metadata_files_name in metadata_files_names:
                    # Create directory name
                    directory_name = metadata_files_name.split('.')[0]
                    # Make the directory according to the metadata file
                    directory_path = os.path.join(subdir_path, directory_name)
                    try:
                        os.mkdir(directory_path)
                    except FileExistsError:
                        already_exist.append(metadata_files_name)
                        print(f"The mixtures from the {metadata_files_name}"
                              f" file already exist, the metadafile"
                              f" will be ignored")

    # Remove the element already used
    for element in already_exist:
        metadata_files_names.remove(element)

    return metadata_files_names


def get_list_from_csv(list_from_csv, desired_type):
    """ Transform a list in the .csv in an actual python list """
    python_list = list_from_csv.strip("['")
    python_list = python_list.strip("']")
    python_list = python_list.replace("'", "")
    python_list = python_list.split(', ')
    if desired_type == 'int':
        for i in range(len(python_list)):
            python_list[i] = int(python_list[i])
    elif desired_type == 'float':
        for i in range(len(python_list)):
            python_list[i] = float(python_list[i])
    return python_list


def loudness_normalize(sources_list, original_loudness, target_loudness_list):
    """ Normalize sources loudness"""
    # Create the list of normalized sources
    normalized_list = []
    for i, source in enumerate(sources_list):
        normalized_list.append(
            pyln.normalize.loudness(source, original_loudness[i],
                                    target_loudness_list[i]))
    return normalized_list


def resample_list(sources_list, freq):
    """ Resample the source list to the desired frequency"""
    # Create the resampled list
    resampled_list = []

    # Resample each source
    for source in sources_list:
        resampled_list.append(resample_poly(source, freq, 16000))

    return resampled_list


def mix(sources_list):
    """ Do the mixing """

    # Initialize mixture
    mixture = np.zeros_like(sources_list[0])
    for source in sources_list:
        mixture += source

    return mixture


def fit_lengths(source_list, mode):
    """ Make the sources to match the target length """
    sources_list_reshaped = []

    # Check the mode
    if mode == 'min':
        target_length = min([len(source) for source in source_list])
        for source in source_list:
            sources_list_reshaped.append(source[:target_length])
    else:
        target_length = max([len(source) for source in source_list])
        for source in source_list:
            sources_list_reshaped.append(
                np.pad(source, (0, target_length - len(source))))

    return sources_list_reshaped


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
