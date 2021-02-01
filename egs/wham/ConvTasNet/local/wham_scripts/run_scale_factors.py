import numpy as np
import os
import pandas as pd
import scipy.io as sio
from scipy.signal import resample_poly
import pyloudnorm as pyln
import soundfile as sf
from constants import SAMPLERATE, MAX_SAMPLE_AMP
from utils import create_wham_mixes, append_or_truncate, read_scaled_wav

# User options
WSJMIX_16K_PATH = '/mm1/wichern/wsj0-mix/2speakers/wav16k'  # Input argument from create_wav_2speakers.m Matlab script
WSJMIX_8K_PATH = '/mm1/wichern/wsj0-mix/2speakers/wav8k' #  Input argument from create_wav_2speakers.m Matlab script
WHAM_NOISE_PATH = '/mm1/wichern/wham_noise'  # The path to the WHAM noise data, output from run_sample_noise.py
# End user options

MIX_PARAM_STUB = os.path.join(WHAM_NOISE_PATH, 'metadata', 'mix_param_meta_{}.csv')
SCALING_MAT = 'scaling.mat'
SCALING_NPZ_OUT = os.path.join(WHAM_NOISE_PATH, 'metadata', 'scaling_{}.npz')

S1_DIR = 's1'
S2_DIR = 's2'

for splt in ['tr', 'cv', 'tt']:
    mix_param_path = MIX_PARAM_STUB.format(splt)
    mix_param_df = pd.read_csv(mix_param_path)
    noise_path = os.path.join(WHAM_NOISE_PATH, splt)
    scaling_out_dict = {}
    for sr_dir, wsj_root in zip(['16k', '8k'], [WSJMIX_16K_PATH, WSJMIX_8K_PATH]):
        scaling_key = 'scaling_{}'.format(sr_dir)

        if sr_dir == '8k':
            loudness_meter = pyln.Meter(8000)
            downsample = True
        else:
            loudness_meter = pyln.Meter(SAMPLERATE)
            downsample = False

        for datalen_dir in ['max', 'min']:
            wsj_path = os.path.join(wsj_root, datalen_dir, splt)
            scaling_path = os.path.join(wsj_path, SCALING_MAT)
            scaling_dict = sio.loadmat(scaling_path)
            scaling_wsjmix = scaling_dict[scaling_key]
            n_utt, n_srcs = scaling_wsjmix.shape
            scaling_noise_wham = np.zeros(n_utt)
            scaling_speech_wham = np.zeros(n_utt)
            speech_start_sample = np.zeros(n_utt)

            print('{} {} dataset, {} split'.format(sr_dir, datalen_dir, splt))

            for i_utt, (utt_id, start_samp_16k, n_end_samp_16k, speaker1_target_snr_db) in \
                    enumerate(mix_param_df.itertuples(index=False, name=None)):

                # read s1 and s2
                s1_path = os.path.join(wsj_path, S1_DIR, utt_id)
                s1_samples, _ = sf.read(s1_path)
                s2_path = os.path.join(wsj_path, S2_DIR, utt_id)
                s2_samples, _ = sf.read(s2_path)

                # read noise
                noise_samples = read_scaled_wav(os.path.join(noise_path, utt_id), scaling_factor=1.0,
                                                downsample_8K=downsample)
                s1_samples, s2_samples, noise_samples = append_or_truncate(s1_samples, s2_samples, noise_samples,
                                                                           datalen_dir, start_samp_16k, downsample)
                # compute gain based on BS.1770 integrated loudness
                speech_level = loudness_meter.integrated_loudness(s1_samples)
                noise_level = loudness_meter.integrated_loudness(noise_samples)
                gain_db = speaker1_target_snr_db + noise_level - speech_level

                # compute mixes
                g = 10 ** (gain_db / 20.)
                s1_samples *= g
                s2_samples *= g
                mix_clean, mix_single, mix_both = create_wham_mixes(s1_samples, s2_samples, noise_samples)

                # check for clipping and fix gains
                max_amp_s1 = np.max(np.abs(s1_samples))
                max_amp_s2 = np.max(np.abs(s2_samples))
                max_amp_nz = np.max(np.abs(noise_samples))
                max_amp_clean = np.max(np.abs(mix_clean))
                max_amp_single = np.max(np.abs(mix_single))
                max_amp_both = np.max(np.abs(mix_both))
                max_amp = np.max([max_amp_s1, max_amp_s2, max_amp_nz, max_amp_clean, max_amp_single, max_amp_both])
                if max_amp > MAX_SAMPLE_AMP:
                    lin_gain = MAX_SAMPLE_AMP / max_amp
                    g *= lin_gain
                else:
                    lin_gain = 1.0

                scaling_speech_wham[i_utt] = g
                scaling_noise_wham[i_utt] = lin_gain

                if (i_utt + 1) % 500 == 0:
                    print('Completed {} of {} utterances'.format(i_utt + 1, len(mix_param_df)))

            scaling_out_dict['scaling_wsjmix_{}_{}'.format(sr_dir, datalen_dir)] = scaling_wsjmix
            scaling_out_dict['scaling_wham_speech_{}_{}'.format(sr_dir, datalen_dir)] = scaling_speech_wham
            scaling_out_dict['scaling_wham_noise_{}_{}'.format(sr_dir, datalen_dir)] = scaling_noise_wham

    scaling_out_dict['speech_start_sample_16k'] = mix_param_df['noise_samples_beginning_16k'].values
    scaling_out_dict['utterance_id'] = mix_param_df['utterance_id'].values
    np.savez(SCALING_NPZ_OUT.format(splt), **scaling_out_dict)
