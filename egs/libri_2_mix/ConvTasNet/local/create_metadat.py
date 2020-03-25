import random
import os
import argparse
import soundfile as sf
import pandas as pd
import numpy as np
import glob


def main(arguments):
    storage_dir = arguments.storage_dir
    dataset_name = arguments.dataset_name
    n_src = arguments.n_src
    create_libri2mix_mixtures_metadata(storage_dir, dataset_name, n_src)


def create_speakers_metadata(storage_dir):
    """ Read metadata from the LibriSpeech dataset and collect infos
    about the speakers """

    # Go to LibriSpeech root directory
    librispeech_root_path = os.path.join(storage_dir, 'LibriSpeech')

    # Read SPEAKERS.TXT and create a dataframe
    speakers_metadata_path = os.path.join(librispeech_root_path,
                                          'SPEAKERS.TXT')
    speakers_metadata = pd.read_csv(speakers_metadata_path, sep="|",
                                    skiprows=11,
                                    error_bad_lines=False, header=0,
                                    names=['Speaker_ID', 'Sex', 'Subset',
                                           'MINUTES', 'NAMES'],
                                    skipinitialspace=True)

    speakers_metadata = speakers_metadata.drop(['MINUTES', 'NAMES'],
                                               axis=1)
    # Delete white space
    for column in ['Sex', 'Subset']:
        speakers_metadata[column] = speakers_metadata[column].str.strip()

    # There is a problem with Speaker_ID = 60 his name contains " | " which is
    # the sperator character... Need to handle this case separately

    speakers_metadata.loc[len(speakers_metadata)] = [60, 'M',
                                                     'train-clean-100']

    return speakers_metadata


def create_librispeech_metadata(storage_dir):
    """ Generate metadata corresponding to downloaded data in LibriSpeech"""

    # Get speakers metadata
    speakers_metadata = create_speakers_metadata(storage_dir)

    # Go to LibriSpeech root directory
    librispeech_root_path = os.path.join(storage_dir, "LibriSpeech")

    # Create metadata directory
    os.mkdir(os.path.join(librispeech_root_path, 'metadata'))
    metadata_directory_path = os.path.join(librispeech_root_path, 'metadata')

    # Possible directories in the original LibriSpeech
    original_librispeech_directories = ['dev-clean', 'dev-other', 'test-clean',
                                        'test-other', 'train-clean-100',
                                        'train-clean-360', 'train-other-500']

    # Actual directories extracted in our LibriSpeech version
    actual_librispeech_directories = list(
        set(next(os.walk(librispeech_root_path))[1]) &
        set(original_librispeech_directories))

    # Go throw each directory and create associated metadata
    for i, directory in enumerate(actual_librispeech_directories):

        # Get the current directory path
        directory_path = os.path.join(librispeech_root_path, directory)

        # Look for .flac files in every subdirectories in the current
        # directory
        sound_paths = glob.glob(os.path.join(directory_path, '**/*.flac'),
                                recursive=True)

        # Create the dataframe corresponding to this directory
        directory_metadata = pd.DataFrame(
            columns=['Speaker_ID', 'Sex', 'Subset', 'Length', 'Origin_Path'])

        # Go throw the sound file list
        for sound_path in sound_paths:
            # Get the sound file path

            relative_path = os.path.relpath(sound_path, librispeech_root_path)

            sound_name = os.path.split(sound_path)[1]

            # Get its length
            Length = len(sf.SoundFile(sound_path))

            # Get the ID of the speaker
            Speaker_ID = sound_name.split('-')[0]

            # Find the Sex according to the speaker ID in the LibriSpeech
            # metadata
            Sex = speakers_metadata[
                speakers_metadata['Speaker_ID'] == int(Speaker_ID)].iat[
                0, 1]

            # Find the subset according to the speaker ID in the LibriSpeech
            # metadata
            Subset = speakers_metadata[
                speakers_metadata['Speaker_ID'] == int(Speaker_ID)].iat[
                0, 2]

            # Add information to the dataframe
            directory_metadata.loc[len(directory_metadata)] = \
                [Speaker_ID, Sex, Subset, Length, relative_path]

        # Sort the dataframe according to ascending Length
        directory_metadata = directory_metadata.sort_values('Length')

        # Write the dataframe in a .csv in the metadata directory
        save_path = os.path.join(metadata_directory_path, directory + '.csv')
        directory_metadata.to_csv(save_path, index=False)


def create_libri2mix_mixtures_metadata(storage_dir, dataset_name, n_src):
    """ Generate metadata for train, test  and validation mixtures """

    # create_librispeech_metadata(storage_dir)

    # Create libri2mix directory
    os.mkdir(os.path.join(storage_dir, dataset_name))
    libri2mix_directory_path = os.path.join(storage_dir, dataset_name)

    # Create metadata directory
    os.mkdir(os.path.join(libri2mix_directory_path, 'metadata'))
    mixtures_metadata_directory_path = os.path.join(libri2mix_directory_path,
                                                    'metadata')

    # Path to Librispeech metadata directory
    librispeech_metadata_directory_path = os.path.join(storage_dir,
                                                       'LibriSpeech/metadata')

    # List metadata files in LibriSpeech
    metadata_file_names = os.listdir(librispeech_metadata_directory_path)

    # Go throw each metadata file and create mixture metadata accordingly
    for metadata_file_name in metadata_file_names:
        # Get the current metadata  file path
        metadata_file_path = os.path.join(librispeech_metadata_directory_path,
                                          metadata_file_name)

        # Open .csv files
        metadata_file = pd.read_csv(metadata_file_path)

        # Create the dataframe corresponding to this directory
        metadata_mixtures_file = pd.DataFrame(
            columns=['Mixture_ID', 'SNR', 'Weight', 'Speaker_ID_list',
                     'Sex_list', 'Path_list', 'Length_list'])

        random.seed(123)

        # Initialize list for pairs sources
        L = []

        # A counter
        c = 0

        # Index
        Index = [i for i in range(len(metadata_file))]

        # Try to create pairs with different speakers end after 20 fails
        while len(Index) > 1 and c < 20:
            couple = random.sample(Index, n_src)

            # Verify that speakers are different
            speaker_list = set([metadata_file.iloc[couple[i]]['Speaker_ID']
                                for i in range(n_src)])

            # If there are duplicates then increment the counter
            if len(speaker_list) != n_src:
                c += 1

            # Else append the combination to L and eras the combination
            # from the available indexes
            else:
                for i in range(n_src):
                    Index.remove(couple[i])
                L.append(couple)
                c = 0

        # Create mixture from pairs and create metadata
        for el in L:

            # Create elements for the dataframe
            Mixtures_ID = ""
            Speaker_ID_list = []
            Sex_list = []
            Length_list = []
            Path_list = []

            for i in range(n_src):
                source = metadata_file.iloc[el[i]]
                Speaker_ID_list.append(source['Speaker_ID'])
                Sex_list.append(source['Sex'])
                Length_list.append(source['Length'])
                Path_list.append(source['Origin_Path'])
                Mixtures_ID += os.path.split(source['Origin_Path'])[1] + '_'

                # TODO find a way to mix
                # s1, _ = sf.read(S1_path, dtype='float32')
                # s2, _ = sf.read(S1_path, dtype='float32')
                # snr = 10 * np.log10(
                #     np.avg(np.square(s1)) / np.avg(np.square(s2)) + 1e-10)
                #
                # weight = np.max(np.abs(s1 + s2))

            snr = 0
            weight = np.array(1)

            # Add information to the dataframe
            metadata_mixtures_file.loc[len(metadata_mixtures_file)] = \
                [Mixtures_ID, snr, weight, Speaker_ID_list, Sex_list,
                 Path_list,
                 Length_list]

        # Write the dataframe in a .csv in the metadata directory
        save_path = os.path.join(mixtures_metadata_directory_path,
                                 metadata_file_name)
        metadata_mixtures_file.to_csv(save_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--storage_dir', type=str, default=None,
                        help='Directory where Librispeech has been downloaded')
    parser.add_argument('--dataset_name', type=str, default=None,
                        help='Name of the directory where the dataset will '
                             ' be created')
    parser.add_argument('--n_src', type=int, default=2,
                        help='Number of sources desired to create the mixture')
    args = parser.parse_args()
    main(args)
