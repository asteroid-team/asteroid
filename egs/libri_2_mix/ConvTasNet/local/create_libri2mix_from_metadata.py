import random
import os
import argparse
import soundfile as sf
import pandas as pd
import numpy as np
import glob


def main(args):

    # Get libri2mix root path
    libri2mix_root_path = args.libri2mix_root_path

    # Generate source
    generate_mixtures(libri2mix_root_path)


def get_longest_sources(s1, s2):

    # Get the max length
    L = [len(s1), len(s2)]
    index = L.index(max(L))

    return index


def mix(s1, s2):

    # Get the longest source
    index = get_longest_sources(s1, s2)

    # Pad the shortest
    if index == 1:
        s1 = np.pad(s1, (int(np.floor((len(s2) - len(s1)) / 2)),
                         int(np.ceil((len(s2) - len(s1)) / 2))))
    else:
        s2 = s1 = np.pad(s2, (int(np.floor((len(s1) - len(s2)) / 2)),
                              int(np.ceil((len(s1) - len(s2)) / 2))))

    return s1 + s2


def generate_mixtures(libri2mix_root_path):

    # Get the metadata directory path
    metadata_dir_path = os.path.join(libri2mix_root_path, 'metadata')

    # Get metadata files name
    metadata_files_names = os.listdir(metadata_dir_path)

    for metadata_files_name in metadata_files_names:

        # Get metadata file path
        metadata_file_path = os.path.join(metadata_dir_path,
                                          metadata_files_name)

        # Read the csv file
        metadata_file = pd.read_csv(metadata_file_path)

        # Make a directory in libri2mix
        directory_path = os.path.join(metadata_dir_path,
                                      metadata_files_name.split('.')[0])

        os.makedirs(directory_path)

        # Go throw the metadata file and generate mixtures
        for index, row in metadata_file.iterrows():
            # Get info about the mixture
            mixtures_path = row['Mixtures_path']
            S1_path = row['S1_path']
            S2_path = row['S2_path']

            # Read the files to make the mixture
            s1, rate = sf.read(S1_path, dtype='float32')
            s2, rate_2 = sf.read(S2_path, dtype='float32')

            # Mix
            mixtures = mix(s1, s2)

            # Save the mixture
            sf.write(mixtures_path, mixtures, rate)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--libri2mix_root_path', type=str, default=None,
                        help='Path to libri2mix root_directory')
    args = parser.parse_args()
    main(args)
