import os
import argparse
import soundfile as sf
import pandas as pd
import glob
from tqdm import tqdm

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--librispeech_dir', type=str, default=None,
                    help='Path to librispeech root directory')


def main(args):
    librispeech_dir = args.librispeech_dir
    # Librispeech metadata directory
    librispeech_md_dir = os.path.join(librispeech_dir, 'metadata')
    os.makedirs(librispeech_md_dir, exist_ok=True)
    create_librispeech_metadata(librispeech_dir, librispeech_md_dir)


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
    sound_paths = glob.glob(os.path.join(dir_path, '**/*.flac'),
                            recursive=True)
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
        subset = speakers_md[speakers_md['speaker_ID'] == int(spk_id)].iat[
            0, 2]
        # Get its length
        length = len(sf.SoundFile(sound_path))
        # Get the sound file relative path
        rel_path = os.path.relpath(sound_path, librispeech_dir)
        # Add information to the dataframe
        dir_md.loc[len(dir_md)] = [spk_id, sex, subset, length, rel_path]
    return dir_md


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
