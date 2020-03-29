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
    dataset_root_path = 'D:/libri3mix'

    # Get the desired frequencies
    freqs = arguments.freqs
    freqs = ['16K', '8K']

    # Get the desired modes
    modes = arguments.modes
    modes = ['min', 'max']

    # Generate source
    generate_mixtures(librispeech_root_path, dataset_root_path, freqs, modes)


def loudness_normalize(sources_list, mode, target_loudness_min_list,
                       target_loudness_max_list):
    meter = pyln.Meter(16000)

    normalized_list = []
    if mode == 'min':
        target_loudness_list = target_loudness_min_list
    else:
        target_loudness_list = target_loudness_max_list

    for i, source in enumerate(sources_list):
        loudness = meter.integrated_loudness(source)
        normalized_list.append(pyln.normalize.loudness(source,
                                                       loudness,
                                                       target_loudness_list[
                                                           i]))

    return normalized_list


def get_list_from_csv(list_from_csv, desired_type):
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


def resample_list(sources_list, freq):
    # Create the resampled list
    resampled_list = []

    # Resample each source
    for source in sources_list:
        resampled_list.append(resample_poly(source, freq, 16000))

    return resampled_list


def fit_lengths(source_list, mode, target_length):
    sources_list_reshaped = []

    if mode == 'min':
        for source in source_list:
            sources_list_reshaped.append(source[:target_length])
    else:
        for source in source_list:
            sources_list_reshaped.append(
                np.pad(source, (0, target_length - len(source))))

    return sources_list_reshaped


def mix(sources_list, mode, weight_min, weight_max):

    if mode == 'min':
        weight = weight_min
    else:
        weight = weight_max

    # Initialize mixture
    mixture = np.zeros_like(sources_list[0])

    for source in sources_list:
        mixture += source

    return mixture * weight


def generate_mixtures(librispeech_root_path, dataset_root_path, freqs, modes):
    # Get the metadata directory path
    metadata_dir_path = os.path.join(dataset_root_path, 'metadata')

    # Get metadata files name
    metadata_files_names = os.listdir(metadata_dir_path)

    # We will check if the mixtures don't already exist we create a list that
    # will contain the metadata files that already have been used. You can also
    # specify metadata files to ignore.
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

    # Remove the element already used
    for element in already_exist:
        metadata_files_names.remove(element)

    for metadata_files_name in metadata_files_names:

        print(f"Creating mixtures from {metadata_files_name}")

        # Get metadata file path
        metadata_file_path = os.path.join(metadata_dir_path,
                                          metadata_files_name)

        # Read the csv file
        metadata_file = pd.read_csv(metadata_file_path)

        # Go throw the metadata file and generate mixtures
        for index, row in metadata_file.iterrows():

            # Get info about the mixture
            mixture_id = row['Mixture_ID']
            weight_min = row['Weight_min']
            weight_max = row['Weight_max']
            sources_paths = get_list_from_csv(row['Path_list'], 'str')
            target_loudness_max_list = get_list_from_csv(
                row['target_loudness_max_list'], 'float')
            target_loudness_min_list = get_list_from_csv(
                row['target_loudness_min_list'], 'float')

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

                    # Check the mode
                    if mode == 'min':
                        target_length = min(
                            [len(source) for source in sources_list])
                    else:
                        target_length = max(
                            [len(source) for source in sources_list])

                    reshaped_sources = fit_lengths(sources_list, mode,
                                                   target_length)

                    # Normalize sources
                    sources_list_norm = loudness_normalize(
                        reshaped_sources, mode,
                        target_loudness_min_list,
                        target_loudness_max_list)

                    # Resample the sources
                    sources_list_resampled = resample_list(sources_list_norm,
                                                           freq)

                    # Do the mixture
                    mixtures = mix(sources_list_resampled, mode,
                                   weight_min, weight_max)

                    # Path to the mode directory
                    mode_path = os.path.join(freq_path, mode)

                    # Get the directory path where the mixture will be saved
                    directory_name = metadata_files_name.split('.')[0]
                    directory_path = os.path.join(mode_path, directory_name)

                    # Save the mixture
                    mixture_path = os.path.join(directory_path, mixture_id +
                                                '.flac')
                    sf.write(mixture_path, mixtures, freq)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
