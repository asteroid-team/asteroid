import os
import numpy as np
import pandas as pd
import argparse
from utils import wavwrite, read_scaled_wav, fix_length


S1_DIR = 's1'
S2_DIR = 's2'
MIX_DIR = 'mix'

FILELIST_STUB = os.path.join('data', 'mix_2_spk_filenames_{}.csv')


def main(wsj_root, wham_noise_root, output_root):

    scaling_npz_stub = os.path.join(wham_noise_root, 'metadata', 'scaling_{}.npz')

    for sr_str in ['16k', '8k']:
        wav_dir = 'wav' + sr_str
        if sr_str == '8k':
            sr = 8000
            downsample = True
        else:
            sr = 16000
            downsample = False

        for datalen_dir in ['max', 'min']:
            for splt in ['tr', 'cv', 'tt']:
                output_path = os.path.join(output_root, wav_dir, datalen_dir, splt)

                s1_output_dir = os.path.join(output_path, S1_DIR)
                os.makedirs(s1_output_dir, exist_ok=True)
                s2_output_dir = os.path.join(output_path, S2_DIR)
                os.makedirs(s2_output_dir, exist_ok=True)
                mix_output_dir = os.path.join(output_path, MIX_DIR)
                os.makedirs(mix_output_dir, exist_ok=True)

                print('{} {} dataset, {} split'.format(wav_dir, datalen_dir, splt))

                # read filenames
                wsjmix_path = FILELIST_STUB.format(splt)
                wsjmix_df = pd.read_csv(wsjmix_path)
                # read scaling file
                scaling_path = scaling_npz_stub.format(splt)
                scaling_npz = np.load(scaling_path, allow_pickle=True)
                wsjmix_key = 'scaling_wsjmix_{}_{}'.format(sr_str, datalen_dir)
                scaling_mat = scaling_npz[wsjmix_key]

                for i_utt, (output_name, s1_path, s2_path) in enumerate(wsjmix_df.itertuples(index=False, name=None)):

                    s1 = read_scaled_wav(os.path.join(wsj_root, s1_path), scaling_mat[i_utt][0], downsample)
                    s2 = read_scaled_wav(os.path.join(wsj_root, s2_path), scaling_mat[i_utt][1], downsample)

                    s1, s2 = fix_length(s1, s2, datalen_dir)
                    mix = s1 + s2
                    wavwrite(os.path.join(mix_output_dir, output_name), mix, sr)
                    wavwrite(os.path.join(s1_output_dir, output_name), s1, sr)
                    wavwrite(os.path.join(s2_output_dir, output_name), s2, sr)

                    if (i_utt + 1) % 500 == 0:
                        print('Completed {} of {} utterances'.format(i_utt + 1, len(wsjmix_df)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str,
                        help='Output directory for writing wsj0-2mix 8 k Hz and 16 kHz datasets.')
    parser.add_argument('--wsj0-root', type=str,
                        help='Path to the folder containing wsj0/')
    parser.add_argument('--wham-noise-root', type=str,
                        help='Path to the downloaded and unzipped wham folder containing metadata/')
    args = parser.parse_args()
    main(args.wsj0_root, args.wham_noise_root, args.output_dir)

