import random
import os
import argparse
import soundfile as sf
import pandas as pd
import numpy as np
import glob


def main(args):
    storage_dir = args.storage_dir
    create_libri2mix_mixtures_metadata(storage_dir)


def create_speakers_metadata(storage_dir='D://'):
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


def create_librispeech_metadata(storage_dir='D://'):
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
                [Speaker_ID, Sex, Subset, Length, sound_path]

        # Sort the dataframe according to ascending Length
        directory_metadata = directory_metadata.sort_values('Length')

        # Write the dataframe in a .csv in the metadata directory
        save_path = os.path.join(metadata_directory_path, directory + '.csv')
        directory_metadata.to_csv(save_path, index=False)


def create_libri2mix_mixtures_metadata(storage_dir='D://'):
    """ Generate metadata for train, test  and validation mixtures """

    create_librispeech_metadata(storage_dir)

    # Create libri2mix directory
    os.mkdir(os.path.join(storage_dir, 'libri2mix'))
    libri2mix_directory_path = os.path.join(storage_dir, 'libri2mix')

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
            columns=['Mixtures_path', 'SNR','Weight', 'S1_Speaker_ID', 'S1_Sex',
                     'S1_path', 'S1_Length', 'S2_Speaker_ID', 'S2_Sex',
                     'S2_path', 'S2_Length'])

        # Initialize list for pairs sources
        L = []

        # A counter
        c = 0

        # Index
        Index = [i for i in range(len(metadata_file))]

        # Try to create pairs with different speakers end after 20 fails
        while len(Index) > 1 and c < 20:
            couple = random.sample(Index, 2)
            if metadata_file.iloc[couple[0]]['Speaker_ID'] == \
                    metadata_file.iloc[couple[1]]['Speaker_ID']:
                c += 1
                continue
            else:
                Index.remove(couple[0])
                Index.remove(couple[1])
                L.append(couple)
                c = 0

        # Create mixture from pairs and create metadata
        for el in L:
            S1 = metadata_file.iloc[el[0]]
            S2 = metadata_file.iloc[el[1]]
            S1_Speaker_ID = S1['Speaker_ID']
            S1_Sex = S1['Sex']
            S1_Length = S1['Length']
            S1_path = S1['Origin_Path']
            S2_Speaker_ID = S2['Speaker_ID']
            S2_Sex = S2['Sex']
            S2_Length = S2['Length']
            S2_path = S2['Origin_Path']
            Mixtures_ID = os.path.split(S1_path)[1] + '_' + \
                          os.path.split(S2_path)[1]

            s1, _ = sf.read(S1_path, dtype='float32')
            s2, _ = sf.read(S1_path, dtype='float32')
            snr = 10 * np.log10(
                np.avg(np.square(s1)) / np.avg(np.square(s2)) + 1e-10)

            weight = np.max(np.abs(s1+s2))

            Mixtures_path = os.path.join(libri2mix_directory_path,
                                         S1['Subset'])
            Mixtures_path = os.path.join(Mixtures_path, Mixtures_ID + '.flac')

            # Add information to the dataframe
            metadata_mixtures_file.loc[len(metadata_mixtures_file)] = \
                [Mixtures_path, snr, weight ,S1_Speaker_ID, S1_Sex, S1_path, S1_Length,
                 S2_Speaker_ID, S2_Sex, S2_path, S2_Length]

        # Write the dataframe in a .csv in the metadata directory
        save_path = os.path.join(mixtures_metadata_directory_path,
                                 metadata_file_name)
        metadata_mixtures_file.to_csv(save_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--storage_dir', type=str, default=None,
                        help='Directory ')
    args = parser.parse_args()
    main(args)
