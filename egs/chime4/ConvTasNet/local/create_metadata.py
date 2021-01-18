import os
import argparse
from glob import glob
import pandas as pd
import numpy as np

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--chime3_dir", type=str, default=None,
    help="Path to CHiME3 root directory"
)

# Set seed for random generation
SEED = 4
np.random.seed(SEED)


def main(args):
    chime3_dir = args.chime3_dir
    create_local_metadata(chime3_dir)


def create_local_metadata(chime3_dir):
    # Get CHiME-3 annotation files
    c3_annot_files = [f for f in glob(
        os.path.join(chime3_dir, "data", "annotations", "*real*.json"))]
    # Get CHiME-4 annotation files
    c4_annot_files = [f for f in glob(
        os.path.join(chime3_dir, "data", "annotations", "*real*.list"))]
    for c3_annot_file_path in c3_annot_files:
        # Read CHiME-3 annotation file
        c3_annot_file = pd.read_json(c3_annot_file_path)
        # subsets : "tr" "dt" "et" origin "real" or "simu"
        subset, origin = os.path.split(c3_annot_file_path)[1].replace('.json',
                                                                      '').split(
            '_')
        # Look for associated CHiME-4 file
        if c3_annot_file_path.replace('.json',
                                      '_1ch_track.list') in c4_annot_files:
            # Read CHiME-4 annotation file
            c4_annot_file = pd.read_csv(
                c3_annot_file_path.replace('.json', '_1ch_track.list'),
                header=None, names=['path'])
        else:
            c4_annot_file = None
        df, df_2 = create_dataframe(chime3_dir, c3_annot_file, c4_annot_file, subset, origin)
        write_dataframe(df, df_2, subset, origin)


def create_dataframe(chime3_dir, c3_annot_file, c4_annot_file, subset, origin):
    # Empty list for DataFrame creation
    row_list = []
    row_list_2 = []
    for row in c3_annot_file.itertuples():
        speaker = row.speaker
        ID = row.wsj_name
        env = row.environment
        # if we are not dealing with tr subset
        if 'tr' not in subset:
            mixture_path = c4_annot_file[
                c4_annot_file['path'].str.contains(ID + '_' + env)].values[
                0][0]
            mixture_path = os.path.join(chime3_dir,
                                        "data/audio/16kHz/isolated/",
                                        mixture_path)

        # if we are dealing with the tr subset
        else:
            channel = np.random.randint(1, 7)
            mixture_path = os.path.join(chime3_dir,
                                        "data/audio/16kHz/isolated/",
                                        subset + '_' + env.lower() + '_' + origin,
                                        speaker + '_' + ID + '_' + f'.CH{channel}'
                                                                   '.wav')
        dot = row.dot
        duration = row.end - row.start
        temp_dict = {'ID': ID, 'subset': subset, 'origin': origin,
                     'env': env,
                     'mixture_path': mixture_path,
                     'duration': duration}
        trans_dict = {'utt_id': ID, 'text':dot}
        row_list.append(temp_dict)
        row_list_2.append(trans_dict)
    df = pd.DataFrame(row_list)
    df_2 = pd.DataFrame(row_list_2)
    return df, df_2


def write_dataframe(df, df2, subset, origin):
    if 'et' in subset:
        subdir = 'test'
    elif 'dt' in subset:
        subdir = 'val'
    else:
        subdir = 'train'
    save_dir = os.path.join('data', subdir)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, origin + '_1_ch_track.csv')
    df.to_csv(save_path, index=False)
    save_path2 = os.path.join(save_dir, origin + '_1_ch_track_annotations.csv')
    df2.to_csv(save_path2, index=False)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
