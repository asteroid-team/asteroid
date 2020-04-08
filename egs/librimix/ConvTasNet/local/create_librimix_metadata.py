import random
import os
import argparse
import soundfile as sf
import pandas as pd
import numpy as np
import glob
import pyloudnorm as pyln
from tqdm import tqdm
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
MIN_LOUDNESS = -33
MAX_LOUDNESS = -25

# A random seed is used for reproducibility
random.seed(123)

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--librispeech_dir', type=str, default=None,
                    help='Path to librispeech root directory')
parser.add_argument('--n_src', type=int, default=2,
                    help='Number of sources desired to create the mixture')
parser.add_argument('--metadata_outdir', type=str, default=None,
                    help='Where librimix metadata files will be stored.')


def main(args):
    librispeech_dir = args.librispeech_dir
    n_src = args.n_src
    # Librispeech metadata directory
    librispeech_md_dir = os.path.join(librispeech_dir, 'metadata')
    os.makedirs(librispeech_md_dir, exist_ok=True)
    # Librimix metadata directory
    md_dir = args.metadata_outdir
    if md_dir is None:
        root = os.path.dirname(librispeech_dir)
        md_dir = os.path.join(root, f'LibriMix/metadata')
    os.makedirs(md_dir, exist_ok=True)
    create_librimix_metadata(librispeech_dir, librispeech_md_dir,
                             md_dir, n_src=n_src)


def create_librispeech_metadata(librispeech_dir, md_dir):
    """ Generate metadata corresponding to downloaded data in LibriSpeech """
    # Get speakers metadata
    speakers_metadata = create_speakers_dataframe(librispeech_dir)
    # If md_dir already exists then check the already generated files
    already_generated_csv = os.listdir(md_dir)
    # Save the already generated files names
    already_generated_csv = [f.strip('.csv') for f in already_generated_csv]
    # Possible directories in the original LibriSpeech
    original_librispeech_dirs = ['dev-clean', 'dev-other', 'test-clean',
                                 'test-other', 'train-clean-100',
                                 'train-clean-360', 'train-other-500']
    # Actual directories extracted in your LibriSpeech version
    actual_librispeech_dirs = (set(next(os.walk(librispeech_dir))[1]) &
                               set(original_librispeech_dirs))
    # Actual directories that haven't already been processed
    not_already_processed_directories = list(set(actual_librispeech_dirs) -
                                             set(already_generated_csv))

    # Go through each directory and create associated metadata
    for ldir in not_already_processed_directories:
        # Generate the dataframe relative to the directory
        dir_metadata = create_librispeech_dataframe(librispeech_dir, ldir,
                                                    speakers_metadata)
        # Filter out files that are shorter than 3s
        number_of_frames = NUMBER_OF_SECONDS * RATE
        dir_metadata = dir_metadata[dir_metadata['length'] >= number_of_frames]
        # Sort the dataframe according to ascending Length
        dir_metadata = dir_metadata.sort_values('length')
        # Write the dataframe in a .csv in the metadata directory
        save_path = os.path.join(md_dir, ldir + '.csv')
        dir_metadata.to_csv(save_path, index=False)


def create_speakers_dataframe(librispeech_dir):
    """ Read metadata from the LibriSpeech dataset and collect infos
    about the speakers """
    print("Reading speakers metadata")
    # Read SPEAKERS.TXT and create a dataframe
    speakers_metadata_path = os.path.join(librispeech_dir, 'SPEAKERS.TXT')
    speakers_metadata = pd.read_csv(speakers_metadata_path, sep="|",
                                    skiprows=11,
                                    error_bad_lines=False, header=0,
                                    names=['speaker_ID', 'sex', 'subset',
                                           'minutes', 'names'],
                                    skipinitialspace=True)

    speakers_metadata = speakers_metadata.drop(['minutes', 'names'],
                                               axis=1)
    # Delete white space
    for column in ['sex', 'subset']:
        speakers_metadata[column] = speakers_metadata[column].str.strip()
    # There is a problem with Speaker_ID = 60 his name contains " | " which is
    # the sperator character... Need to handle this case separately
    speakers_metadata.loc[len(speakers_metadata)] = [60, 'M',
                                                     'train-clean-100']

    return speakers_metadata


def create_librispeech_dataframe(librispeech_dir, subdir, speakers_md):
    """ Generate a dataframe that gather infos about the sound files in a
    LibriSpeech subdirectory """

    print(f"Creating {subdir} metadata file in LibriSpeech/metadata")
    # Get the current directory path
    dir_path = os.path.join(librispeech_dir, subdir)
    # Recursively look for .flac files in current directory
    sound_paths = glob.glob(os.path.join(dir_path, '**/*.flac'), recursive=True)
    # Create the dataframe corresponding to this directory
    dir_md = pd.DataFrame(columns=['speaker_ID', 'sex', 'subset',
                                   'length', 'origin_path'])

    # Go through the sound file list
    for sound_path in tqdm(sound_paths, total=len(sound_paths)):
        # Get the ID of the speaker
        spk_id = os.path.split(sound_path)[1].split('-')[0]
        # Find Sex according to speaker ID in LibriSpeech metadata
        sex = speakers_md[speakers_md['speaker_ID'] == int(spk_id)].iat[0, 1]
        # Find subset according to speaker ID in LibriSpeech metadata
        subset = speakers_md[speakers_md['speaker_ID'] == int(spk_id)].iat[0, 2]
        # Get its length
        length = len(sf.SoundFile(sound_path))
        # Get the sound file relative path
        rel_path = os.path.relpath(sound_path, librispeech_dir)
        # Add information to the dataframe
        dir_md.loc[len(dir_md)] = [spk_id, sex, subset, length, rel_path]
    return dir_md


def create_librimix_metadata(librispeech_dir, in_md_dir, out_md_dir, n_src=2):
    """ Generate LibriMix metadata according to LibriSpeech metadata """
    # Dataset name
    dataset = f'libri{n_src}mix'
    # List metadata files in LibriSpeech
    metadata_files = os.listdir(in_md_dir)

    # If you wish to ignore some metadata files add their name here
    # Example : to_be_ignored = ['dev-other.csv']
    to_be_ignored = []
    # Check if the metadata files in LibriSpeech already have been used
    already_generated = os.listdir(out_md_dir)
    for generated in already_generated:
        if generated.startswith(f"{dataset}") and 'info' not in generated:
            to_be_ignored.append(generated.replace(f"{dataset}_", ""))
            print(f"{generated} already exists in "
                  f"{out_md_dir} it won't be overwritten")
    for element in to_be_ignored:
        metadata_files.remove(element)

    # Go through each metadata file and create metadata accordingly
    for md_file in metadata_files:
        if not md_file.endswith('.csv'):
            print(f"{md_file} is not a csv file, continue.")
            continue
        # Filenames
        save_path = os.path.join(out_md_dir, '_'.join([dataset, md_file]))
        info_name = '_'.join([dataset, md_file.strip('.csv'), 'info']) + '.csv'
        info_save_path = os.path.join(out_md_dir, info_name)
        print(f"Creating {os.path.basename(save_path)} file in {out_md_dir}")

        # Open .csv files from LibriSpeech
        metadata_file = pd.read_csv(os.path.join(in_md_dir, md_file),
                                    engine='python')
        # Filter out files that are shorter than 3s
        num_samples = NUMBER_OF_SECONDS * RATE
        metadata_file = metadata_file[metadata_file['length'] >= num_samples]
        # Create dataframe
        mixtures_md, mixtures_info = create_librimix_df(metadata_file, n_src,
                                                        librispeech_dir)
        # Save csv files
        mixtures_md.to_csv(save_path, index=False)
        mixtures_info.to_csv(info_save_path, index=False)


def create_librimix_df(metadata_file, n_src, librispeech_dir):
    """ Generate librimix dataframe from a LibriSpeech metadata file """
    # Create a dataframe that will be used to generate sources and mixtures
    mixtures_md = pd.DataFrame(columns=['mixture_ID'])
    # Create a dataframe with additional infos.
    mixtures_info = pd.DataFrame(columns=['mixture_ID'])
    # Add columns (depend on the number of sources)
    for i in range(n_src):
        mixtures_md[f"source_{i+1}_path"] = {}
        mixtures_md[f"source_{i+1}_gain"] = {}
        mixtures_info[f"speaker_{i+1}_ID"] = {}
        mixtures_info[f"speaker_{i+1}_sex"] = {}
    # Generate pairs of sources to mix
    pairs = set_pairs(metadata_file, n_src)

    clip_counter = 0
    # For each combination create a new line in the dataframe
    for pair in tqdm(pairs):
        # return infos about the sources, generate sources
        sources_info, sources_list_max = read_sources(metadata_file, pair,
                                                      n_src, librispeech_dir)
        # compute initial loudness, randomize loudness and normalize sources
        loudness, _, sources_list_norm = set_loudness(sources_list_max, n_src)
        # Do the mixture
        mixture_max = mix(sources_list_norm)
        # Check the mixture for clipping and renormalize if necessary
        renormalize_loudness, did_clip = check_for_cliping(mixture_max,
                                                           sources_list_norm)
        clip_counter += int(did_clip)
        # Compute gain
        gain_list = compute_gain(loudness, renormalize_loudness)

        # Add information to the dataframe
        row_mixture, row_info = get_row(sources_info, gain_list, n_src)
        mixtures_md.loc[len(mixtures_md)] = row_mixture
        mixtures_info.loc[len(mixtures_info)] = row_info
    print(f"Among {len(mixtures_md)} mixtures, {clip_counter} clipped.")
    return mixtures_md, mixtures_info


def set_pairs(metadata_file, n_src):
    """ set pairs of sources to make the mixture """
    # Initialize list for pairs sources
    pair_list = []
    # A counter
    c = 0
    # Index of the rows in the metadata file
    index = list(range(len(metadata_file)))

    # Try to create pairs with different speakers end after 200 fails
    while len(index) >= n_src and c < 200:
        couple = random.sample(index, n_src)
        # Verify that speakers are different
        speaker_list = set([metadata_file.iloc[couple[i]]['speaker_ID']
                            for i in range(n_src)])
        # If there are duplicates then increment the counter
        if len(speaker_list) != n_src:
            c += 1
        # Else append the combination to L and erase the combination
        # from the available indexes
        else:
            for i in range(n_src):
                index.remove(couple[i])
            pair_list.append(couple)
            c = 0
    return pair_list


def read_sources(metadata_file, pair, n_src, librispeech_dir):
    # Read lines corresponding to pair
    sources = [metadata_file.iloc[pair[i]] for i in range(n_src)]
    # Get sources info
    speaker_id_list = [source['speaker_ID'] for source in sources]
    sex_list = [source['sex'] for source in sources]
    length_list = [source['length'] for source in sources]
    path_list = [source['origin_path'] for source in sources]
    id_l = [os.path.split(source['origin_path'])[1].strip('.flac')
            for source in sources]
    mixtures_id = "_".join(id_l)

    # Get the longest and shortest source len
    max_length = max(length_list)
    sources_list = []

    # Read the source and compute some info
    for i in range(n_src):
        source = metadata_file.iloc[pair[i]]
        absolute_path = os.path.join(librispeech_dir,
                                     source['origin_path'])
        s, _ = sf.read(absolute_path, dtype='float32')
        sources_list.append(np.pad(s, (0, max_length - len(s))))

    sources_info = {'mixtures_id': mixtures_id,
                    'speaker_id_list': speaker_id_list, 'sex_list': sex_list,
                    'path_list': path_list}
    return sources_info, sources_list


def set_loudness(sources_list, n_src):
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
        loudness_list.append(meter.integrated_loudness(sources_list[i]))
        # Pick a random loudness
        target_loudness = random.uniform(MIN_LOUDNESS, MAX_LOUDNESS)
        # Normalize source to target loudness

        with warnings.catch_warnings():
            # We don't want to pollute stdout, but we don't want to ignore
            # other warnings.
            warnings.simplefilter("ignore")
            src = pyln.normalize.loudness(sources_list[i], loudness_list[i],
                                          target_loudness)
        # If source clips, renormalize
        if np.max(np.abs(src)) >= 1:
            src = sources_list[i] * MAX_AMP / np.max(np.abs(sources_list[i]))
            target_loudness = meter.integrated_loudness(src)
        # Save scaled source and loudness.
        sources_list_norm.append(src)
        target_loudness_list.append(target_loudness)
    return loudness_list, target_loudness_list, sources_list_norm


def mix(sources_list_norm):
    """ Do the mixture for min mode and max mode """
    # Initialize mixture
    mixture_max = np.zeros_like(sources_list_norm[0])
    for i in range(len(sources_list_norm)):
        mixture_max += sources_list_norm[i]
    return mixture_max


def check_for_cliping(mixture_max, sources_list_norm):
    """Check the mixture (mode max) for clipping and re normalize if needed."""
    # Initialize renormalized sources and loudness
    renormalize_loudness = []
    clip = False
    # Recreate the meter
    meter = pyln.Meter(16000)
    # Check for clipping in mixtures
    if np.max(np.abs(mixture_max)) > MAX_AMP:
        clip = True
        weight = MAX_AMP / np.max(np.abs(mixture_max))
    else:
        weight = 1
    # Renormalize
    for i in range(len(sources_list_norm)):
        new_loudness = meter.integrated_loudness(sources_list_norm[i] * weight)
        renormalize_loudness.append(new_loudness)
    return renormalize_loudness, clip


def compute_gain(loudness, renormalize_loudness):
    """ Compute the gain between the original and target loudness"""
    gain = []
    for i in range(len(loudness)):
        delta_loudness = renormalize_loudness[i] - loudness[i]
        gain.append(np.power(10.0, delta_loudness / 20.0))
    return gain


def get_row(sources_info, gain_list, n_src):
    """ Get new row for each mixture/info dataframe """
    row_mixture = [sources_info['mixtures_id']]
    row_info = [sources_info['mixtures_id']]
    for i in range(n_src):
        row_mixture.append(sources_info['path_list'][i])
        row_mixture.append(gain_list[i])
        row_info.append(sources_info['speaker_id_list'][i])
        row_info.append(sources_info['sex_list'][i])
    return row_mixture, row_info


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
