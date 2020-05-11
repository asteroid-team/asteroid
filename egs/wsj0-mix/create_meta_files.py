import os
import argparse
from ipdb import set_trace

from asteroid.data.wsj0_mix import create_wav_id_sample_count_list

parser = argparse.ArgumentParser()
parser.add_argument('base_path', type=str,\
        help='base path containing tr, tt and cv')
parser.add_argument('dest_folder', type=str,\
        help='Path to save the metadata')
args = parser.parse_args()

datasets = ['tr', 'cv', 'tt']

def create_meta_data(base_path, dest_folder):
    for _dataset in datasets:
        ds_base = os.path.join(base_path, _dataset, 'mix')
        meta_dest = os.path.join(dest_folder, _dataset+'.wavid.samples')
        if os.path.exists(meta_dest):
            print('{} already exists. Remove it and rerun'.format(meta_dest))
            exit(1)
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
        create_wav_id_sample_count_list(ds_base, meta_dest)



if __name__ == '__main__':
    set_trace()
    create_meta_data(args.base_path, args.dest_folder)
