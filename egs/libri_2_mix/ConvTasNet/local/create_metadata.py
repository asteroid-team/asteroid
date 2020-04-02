import random
import os
import argparse
import soundfile as sf
import pandas as pd
import numpy as np
import glob
import pyloudnorm as pyln
import warnings

# Some parameters
# eps secures log and division
EPS = 1e-10
# max amplitude in sources and mixtures
MAX_AMP = 0.9
# We will filter out files shorter than that
NUMBER_OF_SECONDS = 3
# In LibriSpeech all the sources are at 16K Hz
RATE = 16000
# We will randomize loudness between this range
MIN_LOUDNESS = -30
MAX_LOUDNESS = -20

# We wil need to catch a user warning and deal with it
warnings.filterwarnings("error")

random.seed(123)

parser = argparse.ArgumentParser()
parser.add_argument('--storage_dir', type=str, default=None,
                    help='Directory where Librispeech has been downloaded')
parser.add_argument('--n_src', type=int, default=2,
                    help='Number of sources desired to create the mixture')
parser.add_argument('--dataset_name', type=str, default=None,
                    help='Name of the directory where the dataset will '
                         ' be created')


def main(arguments):
    storage_dir = arguments.storage_dir
    storage_dir = "D://"

    n_src = arguments.n_src
    n_src = 2

    dataset_name = arguments.dataset_name
    if dataset_name is None:
        dataset_name = f'libri{n_src}mix'

    # Check if the LibriSpeech metadata already exist
    try:
        create_librispeech_metadata(storage_dir)
    except FileExistsError:
        pass
    create_librimix_metadata(storage_dir, dataset_name, n_src)


def create_librispeech_metadata(storage_dir):
    """ Generate metadata corresponding to downloaded data in LibriSpeech """

    # Get speakers metadata
    speakers_metadata = create_speakers_dataframe(storage_dir)
    # Go to LibriSpeech root directory
    librispeech_root_path = os.path.join(storage_dir, "LibriSpeech")
    # Create metadata directory
    os.makedirs(os.path.join(librispeech_root_path, 'metadata'), exist_ok=True)
    metadata_directory_path = os.path.join(librispeech_root_path, 'metadata')
    # If it already exists then check the already generated files
    already_generated_csv = os.listdir(metadata_directory_path)
    # Save the already generated files names
    already_generated_csv = [already_generated.strip('.csv')
                             for already_generated in already_generated_csv]
    # Possible directories in the original LibriSpeech
    original_librispeech_directories = ['dev-clean', 'dev-other', 'test-clean',
                                        'test-other', 'train-clean-100',
                                        'train-clean-360', 'train-other-500']
    # Actual directories extracted in your LibriSpeech version
    actual_librispeech_directories = \
        (set(next(os.walk(librispeech_root_path))[1]) &
         set(original_librispeech_directories))
    # Actual directories that haven't already been processed
    not_already_processed_directories = list(
        set(actual_librispeech_directories) - set(already_generated_csv))

    # Go through each directory and create associated metadata
    for directory in not_already_processed_directories:
        # Generate the dataframe relative to the directory
        directory_metadata = create_librispeech_dataframe(
            librispeech_root_path, directory, speakers_metadata)
        # Filter out files that are shorter than 3s
        number_of_frames = NUMBER_OF_SECONDS * RATE
        directory_metadata = directory_metadata[
            directory_metadata['Length'] >= number_of_frames]
        # Sort the dataframe according to ascending Length
        directory_metadata = directory_metadata.sort_values('Length')
        # Write the dataframe in a .csv in the metadata directory
        save_path = os.path.join(metadata_directory_path, directory + '.csv')
        directory_metadata.to_csv(save_path, index=False)


def create_speakers_dataframe(storage_dir):
    """ Read metadata from the LibriSpeech dataset and collect infos
    about the speakers """

    print("Reading speakers metadata")
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


def create_librispeech_dataframe(librispeech_root_path, directory,
                                 speakers_metadata):
    """ Generate a dataframe that gather infos about the sound files in a
    LibriSpeech subdirectory """

    print(f"Creating {directory} metadata file in LibriSpeech/metadata")
    # Get the current directory path
    directory_path = os.path.join(librispeech_root_path, directory)
    # Look for .flac files in every subdirectories in the current
    # directory
    sound_paths = glob.glob(os.path.join(directory_path, '**/*.flac'),
                            recursive=True)
    # Create the dataframe corresponding to this directory
    directory_metadata = pd.DataFrame(
        columns=['Speaker_ID', 'Sex', 'Subset', 'Length', 'Origin_Path'])

    # Go through the sound file list
    for sound_path in sound_paths:
        # Get the sound file relative path
        relative_path = os.path.relpath(sound_path, librispeech_root_path)
        # Get the sound file name
        sound_name = os.path.split(sound_path)[1]
        # Get its length
        length = len(sf.SoundFile(sound_path))
        # Get the ID of the speaker
        speaker_ID = sound_name.split('-')[0]
        # Find the Sex according to the speaker ID in the LibriSpeech
        # metadata
        sex = speakers_metadata[
            speakers_metadata['Speaker_ID'] == int(speaker_ID)].iat[0, 1]
        # Find the subset according to the speaker ID in the LibriSpeech
        # metadata
        subset = speakers_metadata[
            speakers_metadata['Speaker_ID'] == int(speaker_ID)].iat[0, 2]
        # Add information to the dataframe
        directory_metadata.loc[len(directory_metadata)] = \
            [speaker_ID, sex, subset, length, relative_path]

    return directory_metadata


def create_librimix_metadata(storage_dir, dataset_name, n_src):
    """ Generate metadata for the dataset according to the LibriSpeech
    metadata """

    # Create dataset directory
    os.makedirs(os.path.join(storage_dir, dataset_name), exist_ok=True)
    dataset_directory_path = os.path.join(storage_dir, dataset_name)
    # Create metadata directory
    os.mkdir(os.path.join(dataset_directory_path, 'metadata'))
    mixtures_metadata_directory_path = os.path.join(dataset_directory_path,
                                                    'metadata')
    # Path to Librispeech metadata directory
    librispeech_metadata_directory_path = os.path.join(storage_dir,
                                                       'LibriSpeech/metadata')
    # List metadata files in LibriSpeech
    metadata_file_names = os.listdir(librispeech_metadata_directory_path)
    # If you wish to ignore some metadata files add their name here
    to_be_ignored = []
    for element in to_be_ignored:
        metadata_file_names.remove(element)

    # Go throw each metadata file and create mixture metadata accordingly
    for metadata_file_name in metadata_file_names:
        print(f"Creating {metadata_file_name} "
              f"metadata file in {dataset_name}/metadata")

        # Get the current metadata file path
        metadata_file_path = os.path.join(librispeech_metadata_directory_path,
                                          metadata_file_name)
        # Open .csv files from LibriSpeech
        metadata_file = pd.read_csv(metadata_file_path)
        # Create dataframe
        metadata_mixtures_file, metadata_info_mixtures_file = \
            create_librimix_dataframe(metadata_file, n_src, storage_dir)
        # Write the dataframe in a .csv in the metadata directory
        save_path = os.path.join(mixtures_metadata_directory_path,
                                 'generating_mixture_' + metadata_file_name)
        save_path_2 = os.path.join(mixtures_metadata_directory_path, 'info_' +
                                   metadata_file_name)
        metadata_mixtures_file.to_csv(save_path, index=False)
        metadata_info_mixtures_file.to_csv(save_path_2, index=False)


def create_librimix_dataframe(metadata_file, n_src, storage_dir):
    """ Generate dataset dataframe from a LibriSpeech metadata file """

    # Create a dataframe that will be used to generate sources and mixtures
    metadata_generating_mixtures_file = pd.DataFrame(
        columns=['Mixture_ID'])
    # Create a dataframe that gather information about the sources
    # in the mixtures
    metadata_info_mixtures_file = pd.DataFrame(
        columns=['Mixture_ID'])
    # Add columns they depend on the number of sources
    for i in range(n_src):
        metadata_generating_mixtures_file[f"Source_{i+1}_Path"] = {}
        metadata_generating_mixtures_file[f"Source_{i+1}_Gain"] = {}
        metadata_info_mixtures_file[f"Speaker_{i+1}_ID"] = {}
        metadata_info_mixtures_file[f"Speaker_{i+1}_Sex"] = {}

    # Generate pairs of sources to mix
    pairs = set_pairs(metadata_file, n_src)

    # For each combination create a new line in the dataframe
    for pair in pairs:
        # return infos about the sources, generate sources
        sources_info, sources_list_max = \
            add_sources_info_and_read_sources(metadata_file, pair, n_src,
                                              storage_dir)
        # compute initial loudness, randomize loudness and normalize sources
        loudness, target_loudness_list, sources_list_norm = \
            compute_and_randomize_loudness(sources_list_max, n_src)
        # Do the mixture
        mixture_max = mix(sources_list_norm)
        # Check the mixture for clipping and renormalize if necessary
        renormalize_loudness = \
            check_for_cliping_and_renormalize(mixture_max, sources_list_norm)
        # Compute gain
        gain_list = compute_gain(loudness, renormalize_loudness)

        # Add information to the dataframe
        metadata_generating_mixtures_file, metadata_info_mixtures_file = \
            add_line(sources_info, gain_list,
                     metadata_generating_mixtures_file,
                     metadata_info_mixtures_file, n_src)

    return metadata_generating_mixtures_file, metadata_info_mixtures_file


def set_pairs(metadata_file, n_src):
    """ set pairs of sources to make the mixture """

    # Initialize list for pairs sources
    L = []
    # A counter
    c = 0
    # Index of the rows in the metadata file
    index = list(range(len(metadata_file)))

    # Try to create pairs with different speakers end after 200 fails
    while len(index) >= n_src and c < 200:
        couple = random.sample(index, n_src)
        # Verify that speakers are different
        speaker_list = set([metadata_file.iloc[couple[i]]['Speaker_ID']
                            for i in range(n_src)])
        # If there are duplicates then increment the counter
        if len(speaker_list) != n_src:
            c += 1
        # Else append the combination to L and erase the combination
        # from the available indexes
        else:
            for i in range(n_src):
                index.remove(couple[i])
            L.append(couple)
            c = 0
    return L


def add_sources_info_and_read_sources(metadata_file, pair, n_src, storage_dir):
    # Get LibriSpeech root directory to be able to read the sources
    librispeech_root_directory_path = os.path.join(storage_dir, 'LibriSpeech')

    # Read lines corresponding to pair
    sources = [metadata_file.iloc[pair[i]] for i in range(n_src)]
    # Get sources info
    speaker_id_list = [source['Speaker_ID'] for source in sources]
    sex_list = [source['Sex'] for source in sources]
    length_list = [source['Length'] for source in sources]
    path_list = [source['Origin_Path'] for source in sources]
    id_l = [os.path.split(source['Origin_Path'])[1].strip('.flac')
            for source in sources]
    mixtures_id = "_".join(id_l)

    # Get the longest and shortest source len
    max_length = max(length_list)
    sources_list = []

    # Read the source and compute some info
    for i in range(n_src):
        source = metadata_file.iloc[pair[i]]
        relative_path = source['Origin_Path']
        absolute_path = os.path.join(librispeech_root_directory_path,
                                     relative_path)
        s, _ = sf.read(absolute_path, dtype='float32')
        s_max = np.pad(s, (0, max_length - len(s)))
        sources_list.append(s_max)

    sources_info = {'mixtures_id': mixtures_id,
                    'speaker_id_list': speaker_id_list, 'sex_list': sex_list,
                    'path_list': path_list}

    return sources_info, sources_list


def compute_and_randomize_loudness(sources_list, n_src):
    """ Compute original loudness and normalise them randomly """

    # Initialize loudness
    loudness_list = []
    # In LibriSpeech all sources are at 16KHz hence the meter
    meter = pyln.Meter(RATE)
    # Randomize sources loudness
    target_loudness_list = []
    sources_list_norm = []

    # Normalize loudness
    for i in range(n_src):

        # Compute initial loudness
        loudness_list.append(
            meter.integrated_loudness(sources_list[i]))
        # Pick a random loudness
        target_loudness = random.uniform(MIN_LOUDNESS, MAX_LOUDNESS)

        try:
            # Normalize loudness
            sources_list_norm.append(pyln.normalize.loudness(sources_list[i],
                                                             loudness_list[i],
                                                             target_loudness))
            # Save the loudness
            target_loudness_list.append(target_loudness)
        # Catch user warning and normalize file to max amp
        except UserWarning:
            # Normalize to max amp
            sources_list_norm.append(
                sources_list[i] * MAX_AMP / np.max(np.abs(
                    sources_list[i])))
            # Save the loudness
            target_loudness_list.append(
                meter.integrated_loudness(sources_list_norm[i]))

    return loudness_list, target_loudness_list, sources_list_norm


def mix(sources_list_norm):
    """ Do the mixture for min mode and max mode """
    # Initialize mixture
    mixture_max = np.sum(sources_list_norm)

    return mixture_max


def check_for_cliping_and_renormalize(mixture_max, sources_list_norm):
    """ Check the mixture (mode max) for clipping and re normalize accordingly
    """
    # Initialize renormalized sources and loudness
    renormalize_loudness = []
    renormalize_sources = []
    # Recreate the meter
    meter = pyln.Meter(16000)
    # Check for clipping in mixtures
    if np.max(np.abs(mixture_max)) > MAX_AMP:
        weight = MAX_AMP / np.max(np.abs(mixture_max))

    else:
        weight = 1

    # Renormalize
    for i in range(len(sources_list_norm)):
        renormalize_sources.append(sources_list_norm[i] * weight)
        renormalize_loudness.append(
            meter.integrated_loudness(renormalize_sources[i]))

    return renormalize_loudness


def compute_gain(loudness, renormalize_loudness):
    """ Compute the gain according to the original and target loudness"""
    gain = []
    for i in range(len(loudness)):
        delta_loudness = renormalize_loudness[i] - loudness[i]
        gain.append(np.power(10.0, delta_loudness / 20.0))
    return gain


def add_line(sources_info, gain_list, metadata_generating_mixtures_file,
             metadata_info_mixtures_file, n_src):
    """ Add a new line to each dataframe """

    row_mixture = [sources_info['mixtures_id']]
    row_info = [sources_info['mixtures_id']]
    for i in range(n_src):
        row_mixture.append(sources_info['path_list'][i])
        row_mixture.append(gain_list[i])
        row_info.append(sources_info['speaker_id_list'][i])
        row_info.append(sources_info['sex_list'][i])

    metadata_generating_mixtures_file.loc[
        len(metadata_generating_mixtures_file)] = row_mixture
    metadata_info_mixtures_file.loc[
        len(metadata_info_mixtures_file)] = row_info

    return metadata_generating_mixtures_file, metadata_info_mixtures_file


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
