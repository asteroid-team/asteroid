import os
import argparse
import soundfile as sf
import pandas as pd
import numpy as np
from scipy.signal import resample_poly
from tqdm import tqdm

# eps secures log and division
EPS = 1e-10

parser = argparse.ArgumentParser()
parser.add_argument('--librispeech_dir', type=str, required=True,
                    help='Path to librispeech root directory')
parser.add_argument('--metadata_dir', type=str, required=True,
                    help='Path to the LibriMix metadata directory')
parser.add_argument('--librimix_outdir', type=str, default=None,
                    help='Path to the desired dataset root directory')
parser.add_argument('--n_src', type=int, default=2,
                    help='Number of sources in mixtures')
parser.add_argument('--freqs', nargs='+', default=['8k'],
                    help='--freqs 16k 8k will create 2 directories wav8k '
                         'and wav16k')
parser.add_argument('--modes', nargs='+', default=['min'],
                    help='--modes min max will create 2 directories in '
                         'each freq directory')


def main(args):
    print(args)
    # Get librispeech root path
    librispeech_dir = args.librispeech_dir
    # Get Metadata directory
    metadata_dir = args.metadata_dir
    # Get LibriMix root path
    librimix_outdir = args.librimix_outdir
    if librimix_outdir is None:
        librimix_outdir = os.path.dirname(metadata_dir)
    librimix_outdir = os.path.join(librimix_outdir, f'Libri{args.n_src}Mix')
    # Get the desired frequencies
    freqs = args.freqs
    freqs = [freq.lower() for freq in freqs]
    # Get the desired modes
    modes = args.modes
    modes = [mode.lower() for mode in modes]

    create_librimix(librispeech_dir, librimix_outdir, metadata_dir,
                    freqs=freqs, n_src=args.n_src, modes=modes)


def create_librimix(in_dir, out_dir, metadata_dir, freqs=None, n_src=2,
                    modes=None):
    """ Generate sources mixtures and saves them in out_dir"""
    # Get metadata files
    md_filename_list = [file for file in os.listdir(metadata_dir)
                        if 'info' not in file]
    # Create all parts of librimix
    for md_filename in md_filename_list:
        csv_path = os.path.join(metadata_dir, md_filename)
        process_metadata_file(csv_path, freqs, n_src, in_dir, out_dir,
                              modes=modes)


def process_metadata_file(csv_path, freqs, n_src, in_dir, out_dir,
                          modes=None):
    """ Process a metadata generation file to create sources and mixtures"""
    md_file = pd.read_csv(csv_path)
    for freq in freqs:
        # Get the frequency directory path
        freq_path = os.path.join(out_dir, 'wav' + freq)
        # Transform freq = "16k" into 16000
        freq = int(freq.strip('k')) * 1000

        for mode in modes:
            # Path to the mode directory
            mode_path = os.path.join(freq_path, mode)
            # Create two dataframes
            metrics_df = create_empty_metrics_metadata(n_src)
            mixture_df = create_empty_mixture_metadata(n_src)
            # Directory where the mixtures and sources will be stored
            dir_name = os.path.basename(csv_path).replace(
                f'libri{n_src}mix_', '').replace('-clean', '').replace(
                '.csv', '')
            dir_path = os.path.join(mode_path, dir_name)
            # If the files already exist then continue the loop
            if os.path.isdir(dir_path):
                print(f"Directory {dir_path} already exist. "
                      f"Files won't be overwritten")
                continue

            print(f"Creating mixtures and sources from {csv_path} "
                  f"in {dir_path}")
            # Subset metadata path
            subset_metadata_path = os.path.join(mode_path, 'metadata')
            os.makedirs(subset_metadata_path, exist_ok=True)
            # Create directories for mixture and sources
            source_sav_dir = [os.path.join(dir_path, f's{i+1}')
                              for i in range(n_src)]
            mix_save_dir = os.path.join(dir_path, 'mix')
            for subdir in [mix_save_dir] + source_sav_dir:
                os.makedirs(subdir, exist_ok=True)

            # Go through the metadata file and generate mixtures
            for index, row in tqdm(md_file.iterrows(), total=len(md_file)):
                # Get sources and mixture infos
                mix_id, gain_list, sources = read_sources(row, n_src, in_dir)
                # Transform sources
                transformed_sources = transform_sources(sources, freq, mode,
                                                        gain_list)
                # Mix sources
                mixture = mix(transformed_sources)
                length = len(mixture)
                # Compute SNR
                snr_list = compute_snr_list(mixture, transformed_sources)

                # Write sources and mixtures and save their path
                abs_source_path_list = []
                ex_filename = mix_id + '.wav'
                abs_mix_path = os.path.join(mix_save_dir, ex_filename)
                sf.write(abs_mix_path, mixture, freq)
                for src, src_dir in zip(transformed_sources, source_sav_dir):
                    abs_save_path = os.path.join(src_dir, ex_filename)
                    sf.write(abs_save_path, src, freq)
                    abs_source_path_list.append(abs_save_path)

                # Add line to the dataframes
                add_to_metrics_metadata(metrics_df, mix_id, snr_list)
                add_to_mixture_metadata(mixture_df, mix_id,
                                        abs_mix_path,
                                        abs_source_path_list, length)
            # Save the metadata
            save_path_mixture = os.path.join(subset_metadata_path,
                                             'mixture_' + dir_name + '.csv')
            save_path_metrics = os.path.join(subset_metadata_path,
                                             'metrics_' + dir_name + '.csv')
            mixture_df.to_csv(save_path_mixture, index=False)
            metrics_df.to_csv(save_path_metrics, index=False)


def create_empty_metrics_metadata(n_src):
    """ Create the metrics dataframe"""
    metrics_dataframe = pd.DataFrame()
    metrics_dataframe['mixture_ID'] = {}
    for i in range(n_src):
        metrics_dataframe[f"source_{i + 1}_SNR"] = {}
    return metrics_dataframe


def create_empty_mixture_metadata(n_src):
    """ Create the mixture dataframe"""
    mixture_dataframe = pd.DataFrame()
    mixture_dataframe['mixture_ID'] = {}
    mixture_dataframe['mixture_path'] = {}
    for i in range(n_src):
        mixture_dataframe[f"source_{i + 1}_path"] = {}
    mixture_dataframe['length'] = {}
    return mixture_dataframe


def read_sources(row, n_src, librispeech_dir):
    """ Get sources and info to mix the sources """
    # Get info about the mixture
    mixture_id = row['mixture_ID']
    sources_path_list = get_list_from_csv(row, 'source_path', n_src)
    gain_list = get_list_from_csv(row, 'source_gain', n_src)
    sources_list = []

    # Read the files to make the mixture
    for sources_path in sources_path_list:
        sources_path = os.path.join(librispeech_dir,
                                    sources_path)
        source, _ = sf.read(sources_path, dtype='float32')
        sources_list.append(source)
    return mixture_id, gain_list, sources_list


def get_list_from_csv(row, column, n_src):
    """ Transform a list in the .csv in an actual python list """
    python_list = []
    for i in range(n_src):
        current_column = column.split('_')
        current_column.insert(1, str(i + 1))
        current_column = '_'.join(current_column)
        python_list.append(row[current_column])
    return python_list


def transform_sources(sources_list, freq, mode, gain_list):
    """ Transform libriSpeech sources to librimix """
    # Normalize sources
    sources_list_norm = loudness_normalize(sources_list, gain_list)
    # Resample the sources
    sources_list_resampled = resample_list(sources_list_norm, freq)
    # Reshape sources
    reshaped_sources = fit_lengths(sources_list_resampled, mode)
    return reshaped_sources


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


def mix(sources_list):
    """ Do the mixing """
    # Initialize mixture
    mixture = np.zeros_like(sources_list[0])
    for source in sources_list:
        mixture += source
    return mixture


def compute_snr_list(mixture, sources_list):
    """Compute the SNR on the mixture mode min"""
    snr_list = []
    # Compute SNR for min mode
    for i in range(len(sources_list)):
        noise_min = mixture - sources_list[i]
        snr_list.append(snr_xy(sources_list[i], noise_min))
    return snr_list


def snr_xy(x, y):
    return 10 * np.log10(np.mean(x**2) / (np.mean(y**2) + EPS) + EPS)


def add_to_metrics_metadata(metrics_df, mixture_id, snr_list):
    """ Add a new line to metrics_df"""
    row_metrics = [mixture_id] + snr_list
    metrics_df.loc[len(metrics_df)] = row_metrics


def add_to_mixture_metadata(mix_df, mix_id, abs_mix_path, abs_sources_path,
                            length):
    """ Add a new line to mixture_df """
    row_mixture = [mix_id, abs_mix_path] + abs_sources_path + [length]
    mix_df.loc[len(mix_df)] = row_mixture


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
