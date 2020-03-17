import os
import glob
import random
import soundfile as sf
from scipy.signal import resample_poly
import argparse
import pandas as pd
import numpy as np


def main(args):
    in_root = args.in_dir
    out_root = args.out_dir
    freq = args.freq

    # Generate raw metadata
    metadata_librispeech = create_librispeech_metadata(in_root)

    # Generate sources
    generate_sources(args)

    # Generate mixtures and metadata
    generate_mixtures_metadata(out_root, freq, metadata_librispeech)


def create_librispeech_metadata(in_root='D://'):
    """ Read metadata from the origanl LibriSpeech dataset and collect infos
    about the speakers """

    # Go to LibriSpeech directory
    in_root = os.path.join(in_root, 'LibriSpeech')

    # Read SPEAKERS.TXT and create dataframe
    metadata_librispeech_path = os.path.join(in_root, 'SPEAKERS.TXT')
    metadata_librispeech = pd.read_csv(metadata_librispeech_path, sep="|",
                                       skiprows=11,
                                       error_bad_lines=False, header=0,
                                       names=['Speaker_ID', 'Sex', 'SUBSET',
                                              'MINUTES', 'NAMES'],
                                       skipinitialspace=True)

    metadata_librispeech = metadata_librispeech.drop(['MINUTES', 'NAMES'],
                                                     axis=1)
    return metadata_librispeech


def mix(metadata_dir, sources_dir, destination_dir, metadata):
    """ Mix 2 sources from a directory and write the mixture in destination.
     In the meantime create metadata associated with each mixture """

    # Set seed
    random.seed(123)

    # list sources in sources directory
    sources = os.listdir(sources_dir)

    # Initialize list for pairs sources
    L = []

    # A counter
    c = 0

    # Create mixtures meatdata
    metadata_target = pd.DataFrame(
        columns=['S1_Speaker_ID', 'S1_Sex', 'S2_Speaker_ID',
                 'S2_Sex', 'Gain'])

    # Try to create pairs with different speakers end after 20 fails
    while len(sources) > 1 and c < 20:
        couple = random.sample(sources, 2)
        if couple[0].split('-')[0] == couple[1].split('-')[0]:
            c += 1
            continue
        else:
            sources.remove(couple[0])
            sources.remove(couple[1])
            L.append(couple)
            c = 0

    # Create mixture from pairs and create metadata
    for el in L:
        S1_Speaker_ID = el[0].split('-')[0]
        S2_Speaker_ID = el[1].split('-')[0]
        S1_Sex = metadata[metadata['Speaker_ID'] == int(S1_Speaker_ID)].iat[
            0, 1]
        S2_Sex = metadata[metadata['Speaker_ID'] == int(S2_Speaker_ID)].iat[
            0, 1]
        s1, rate_1 = sf.read(os.path.join(sources_dir, el[0]), dtype="float32")
        s2, rate_2 = sf.read(os.path.join(sources_dir, el[1]), dtype="float32")

        mixture = s1 + s2
        g = np.max(np.abs(mixture))
        mixture /= g
        Gain = g
        sf.write(os.path.join(destination_dir, el[0] + "_" + el[1] + ".flac"),
                 mixture, rate_1)

        metadata_target.loc[len(metadata_target)] = [S1_Speaker_ID, S1_Sex,
                                                     S2_Speaker_ID, S2_Sex,
                                                     Gain]

    save_path = os.path.join(metadata_dir,
                             os.path.basename(destination_dir) + '.csv')

    metadata_target.to_csv(save_path)


def generate_sources(args):
    """ Take the raw files from LibriSpeech and create sources for Libri2mix.
    The raw files are dropped if their length is under 4 seconds, otherwise,
    they are cut at 4 seconds.

    """
    # Directory containing the extracted LibriSpeech datatset
    in_root = args.in_dir

    # Directory where the Libri2mix dataset will be created
    out_root = args.out_dir

    # Create Libri2mix directory in output directory
    os.mkdir(os.path.join(out_root, 'libri2mix'))
    out_root = os.path.join(out_root, 'libri2mix')

    # The sampling frequencies of the desired sources
    freqs = args.freq

    # Names of the folder containing the sources. This follows
    # libri2mix_dataset.py implementation
    targets = ['cv', 'test', 'train']  # This is sorted alphabetically

    # Create directories in libri2mix according to parameters
    for freq in freqs:

        # Create directory in Libri2mix corresponding to
        # the sampling frequency
        path_freq = os.path.join(out_root, freq)
        os.mkdir(path_freq)

        # Create a source directory inside the freq directory
        path_source = os.path.join(path_freq, 'sources')
        os.mkdir(path_source)

        for target in targets:
            # Create the directory where the files will be dropped
            target_path = os.path.join(path_source, target)
            os.mkdir(target_path)

    # Path to LibriSpeech root directory
    in_root = os.path.join(in_root, 'LibriSpeech')

    # Get only the directory names in librispeech directory that correspond to
    # train test and cv
    dirs = next(os.walk(in_root))[1]

    # Sort the list alphabetically to match targets
    dirs = sorted(dirs)

    # Remove Raw_metadata directory
    del dirs[1]

    # For each directory in LibriSpeech0 create the corresponding one
    # in libri2mix for each desired sampling frequency
    for i, direc in enumerate(dirs):

        # Path to the current directory
        path_dir = os.path.join(in_root, direc)

        # Look for .flac files in every subdirectories in the current
        # directory
        flac_files = os.path.join(path_dir, '**/*.flac')
        flac_list = glob.glob(flac_files, recursive=True)

        # Create sources of 4 seconds
        for file in flac_list:

            # Get the file name
            file_name = os.path.split(file)[1]

            # Read the original flac file
            data, rate = sf.read(file, frames=16000 * 4, dtype='float32')

            # Files under 4 seconds are dropped
            if len(data) < 64000:
                pass

            # Otherwise, they are resampled to the desired frequencies
            # and dropped in the corresponding folder
            else:
                for freq in freqs:
                    # Create target path
                    target_path = os.path.join(out_root, freq)
                    target_path = os.path.join(target_path, 'sources')
                    target_path = os.path.join(target_path, targets[i])

                    # Resample data
                    rate_resampled = int(freq.strip('K')) * 1000
                    data_resampled = resample_poly(data, rate_resampled, rate)

                    # Write data
                    sf.write(os.path.join(target_path, file_name),
                             data_resampled, rate_resampled)


def generate_mixtures_metadata(out_root, freqs, metadata):
    # Create mixtures
    for freq in freqs:

        # Go to libri2mix freq directory
        path = os.path.join(out_root, 'libri2mix')
        path = os.path.join(path, freq)

        # Create mixture directory
        mixtures_path = os.path.join(path, 'mixtures')
        os.mkdir(os.path.join(path, 'mixtures'))

        # Create metadata folder
        os.mkdir(os.path.join(mixtures_path, 'metadata'))
        metadata_dir = os.path.join(mixtures_path, 'metadata')

        # Get sources directory
        sources_path = os.path.join(path, 'sources')
        dirs = sorted(os.listdir(sources_path))

        for i, direc in enumerate(dirs):
            # Get subset in source directory
            path_dir = os.path.join(sources_path, direc)

            # Create subset in mixture directory
            target_path = os.path.join(mixtures_path, direc)
            os.mkdir(target_path)

            # Make mixtures and metadata
            mix(metadata_dir, path_dir, target_path, metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, default=None,
                        help='Directory path of LibriSpeech')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Directory path to put output files')
    parser.add_argument('--freq', type=list, default=['16K', '8K'],
                        help="""Sampling frequencies desired, format = nK where
                        n is the frequency in Khz, max = 16K""")
    args = parser.parse_args()
    main(args)
