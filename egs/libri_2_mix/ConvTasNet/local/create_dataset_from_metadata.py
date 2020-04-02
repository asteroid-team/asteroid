import os
import argparse
import soundfile as sf
import pandas as pd
import numpy as np
from scipy.signal import resample_poly

# eps secures log and division
EPS = 1e-10

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
    librispeech_root_path = "D://LibriSpeech"
    # Get dataset root path
    dataset_root_path = arguments.dataset_root_path
    dataset_root_path = "D://libri2mix"
    # Get the desired frequencies
    freqs = arguments.freqs
    freqs = [freq.upper() for freq in freqs]
    freqs = ['16K', '8K']
    # Get the desired modes
    modes = arguments.modes
    modes = [mode.lower() for mode in modes]
    modes = ['min', 'max']
    # Generate source
    transform_librispeech(librispeech_root_path, dataset_root_path,
                          freqs, modes)


def transform_librispeech(
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
        # Compute n_src
        n_src = (len(metadata_file.columns) - 1) // 2

        process_metadata_file(metadata_files_name, metadata_file,
                              freqs, modes, n_src,
                              librispeech_root_path, dataset_root_path)


def process_metadata_file(metadata_files_name, metadata_file, freqs,
                          modes, n_src, librispeech_root_path,
                          dataset_root_path):
    for freq in freqs:

        # Get the frequency directory path
        freq_path = os.path.join(dataset_root_path, freq)

        # Transform freq = "16K" into 16000
        freq = int(freq.strip('K')) * 1000

        for mode in modes:

            # Path to the mode directory
            mode_path = os.path.join(freq_path, mode)

            # Create two dataframes
            metrics_df = create_metrics_metadata(n_src)
            mixture_df = create_mixture_metadata(n_src)

            # Go through the metadata file and generate mixtures
            for index, row in metadata_file.iterrows():
                # Get sources and mixture infos
                mixture_id, gain_list, sources_list = \
                    get_mixture_info_and_read_sources(row, n_src,
                                                      librispeech_root_path)
                # Transform sources
                transformed_source_list = transform_sources(sources_list,
                                                            freq, mode,
                                                            gain_list)
                # Mix sources
                mixture = mix(transformed_source_list)
                # Compute SNR
                snr_list = compute_snr(mixture, transformed_source_list)
                # Write sources and mixtures and save their path
                absolute_mixture_path, absolute_source_path_list = \
                    write_sources_and_mixture(freq, transformed_source_list,
                                              mixture, mixture_id, mode_path,
                                              metadata_files_name)
                # Add line to the dataframes
                add_to_metrics_metadata(metrics_df, mixture_id, snr_list)
                add_to_mixture_metadata(mixture_df, mixture_id,
                                        absolute_mixture_path,
                                        absolute_source_path_list)

            directory_name = metadata_files_name.replace(
                'generating_mixture_', "").split('.')[0]
            # Subset metadata path
            subset_metadata_path = os.path.join(mode_path, 'metadata')
            # Save the metadata
            save_path_mixture = os.path.join(subset_metadata_path, 'mixture_'
                                             + directory_name + '.csv')
            save_path_metrics = os.path.join(subset_metadata_path, 'metrics_'
                                             + directory_name + '.csv')
            mixture_df.to_csv(save_path_mixture)
            metrics_df.to_csv(save_path_metrics)


def add_to_metrics_metadata(metrics_df, mixture_id, snr_list):
    """ add a new line to metrics_df"""
    row_metrics = [mixture_id] + snr_list
    metrics_df.loc[len(metrics_df)] = row_metrics


def add_to_mixture_metadata(mixture_df, mixture_id, absolute_mixture_path,
                            absolute_source_path_list):
    """ add a new line to mixture_df """
    row_mixture =\
        [mixture_id, absolute_mixture_path] + absolute_source_path_list
    mixture_df.loc[len(mixture_df)] = row_mixture


def write_sources_and_mixture(freq, transformed_source_list, mixture,
                              mixture_id, mode_path, metadata_files_name):
    """ Write sources and mixtures and return their absolute path"""
    # Path to the sources directory
    sources_path = os.path.join(mode_path, 'sources')
    # Path to the mixtures directory
    mixtures_path = os.path.join(mode_path, 'mixtures')
    # Get the directory path where the mixture will be saved
    directory_name = metadata_files_name.replace(
        'generating_mixture_', "").split('.')[0]
    directory_mixture_path = os.path.join(mixtures_path,
                                          directory_name)
    directory_source_path = os.path.join(sources_path,
                                         directory_name)
    absolute_source_path_list = []
    for i, source in enumerate(transformed_source_list):
        source_id = mixture_id.split('_')[i]
        source_path = os.path.join(directory_source_path,
                                   source_id + '.wav')
        absolute_source_path_list.append(source_path)
        sf.write(source_path, source, freq)
    # Save the mixture
    absolute_mixture_path = os.path.join(directory_mixture_path,
                                         mixture_id + '.wav')
    sf.write(absolute_mixture_path, mixture, freq)
    return absolute_mixture_path, absolute_source_path_list


def transform_sources(sources_list, freq, mode, gain_list):
    """ Transform libriSpeech sources to librimix """
    # Normalize sources
    sources_list_norm = loudness_normalize(
        sources_list, gain_list)
    # Resample the sources
    sources_list_resampled = resample_list(sources_list_norm,
                                           freq)
    # Reshape sources
    reshaped_sources = fit_lengths(sources_list_resampled,
                                   mode)
    return reshaped_sources


def get_mixture_info_and_read_sources(row, n_src, librispeech_root_path):

    # Get info about the mixture
    mixture_id = row['Mixture_ID']
    sources_path_list = get_list_from_csv(row, 'Source_Path',
                                          n_src)
    gain_list = get_list_from_csv(row, 'Source_Gain', n_src)
    sources_list = []

    # Read the files to make the mixture
    for sources_path in sources_path_list:
        sources_path = os.path.join(librispeech_root_path,
                                    sources_path)
        source, _ = sf.read(sources_path, dtype='float32')
        sources_list.append(source)
    return mixture_id, gain_list, sources_list


def create_directories(dataset_root_path, metadata_dir_path, freqs, modes):
    """ Create directories in librimix"""
    # Get metadata files name
    metadata_files_names = os.listdir(metadata_dir_path)

    # We will check if the mixtures and sources  don't already exist we create
    # a list that will contain the metadata files that already have been used.
    # You can also specify metadata files to ignore.
    already_exist = []

    # Ignore info files
    for metadata_files_name in metadata_files_names:
        if metadata_files_name.startswith('info'):
            already_exist.append(metadata_files_name)

    # Remove the element already used
    for element in already_exist:
        metadata_files_names.remove(element)

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
                os.makedirs(os.path.join(mode_path, 'metadata'), exist_ok=True)
                for metadata_files_name in metadata_files_names:
                    # Create directory name
                    directory_name = metadata_files_name.replace(
                        "generating_mixture_", "").split('.')[0]
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


def get_list_from_csv(row, column, n_src):
    """ Transform a list in the .csv in an actual python list """
    python_list = []
    for i in range(n_src):
        current_column = column.split('_')
        current_column.insert(1, str(i + 1))
        current_column = '_'.join(current_column)
        python_list.append(row[current_column])

    return python_list


def loudness_normalize(sources_list, gain_list):
    """ Normalize sources loudness"""
    # Create the list of normalized sources
    normalized_list = []
    for i, source in enumerate(sources_list):
        normalized_list.append(source * gain_list[i])
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


def compute_snr(mixture, sources_list):
    """Compute the SNR on the mixture mode min"""
    snr_list = []

    # Compute SNR for min mode
    for i in range(len(sources_list)):
        noise_min = mixture - sources_list[i]
        snr_list.append(10 * np.log10(
            np.mean(np.square(sources_list[i]))
            / (np.mean(np.square(noise_min)) + EPS) + EPS))

    return snr_list


def create_metrics_metadata(n_src):
    metrics_dataframe = pd.DataFrame()
    metrics_dataframe['Mixture_ID'] = {}
    for i in range(n_src):
        metrics_dataframe[f"Source_{i + 1}_SNR"] = {}
    return metrics_dataframe


def create_mixture_metadata(n_src):
    mixture_dataframe = pd.DataFrame()
    mixture_dataframe['Mixture_ID'] = {}
    mixture_dataframe['Mixture_Path'] = {}
    for i in range(n_src):
        mixture_dataframe[f"Source_{i + 1}_Path"] = {}
    return mixture_dataframe


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
