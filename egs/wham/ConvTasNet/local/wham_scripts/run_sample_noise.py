import numpy as np
import pandas as pd
import os
import soundfile as sf
from noisesampler import NoiseSampler
from constants import SAMPLERATE

# User options
WSJMIX_16K_MAX_PATH = '/mm1/wichern/wsj0-mix/2speakers/wav16k/max'
RAW_NOISE_16K_PATH = '/mm1/wichern/whisper_noise_download_16K'
DENOISE_16K_PATH = '/mm1/wichern/whisper_noise_RX_DE_16K'
OUTPUT_PATH = '/mm1/wichern/wham_noise'
# End user options

NOISE_SPLIT_CSV = os.path.join('data', 'file_splits.csv')
FILELIST_STUB = os.path.join('data', 'mix_2_spk_filenames_{}.csv')
SPLIT_NAMES = {'Train': 'tr', 'Valid': 'cv', 'Test': 'tt'}

SEED = 17
np.random.seed(SEED)

METADATA_DIR = os.path.join(OUTPUT_PATH, 'metadata')
os.makedirs(METADATA_DIR, exist_ok=True)


def get_wsjmix_length(utt_id, split):
    mix_path = os.path.join(WSJMIX_16K_MAX_PATH, split, 'mix', utt_id)
    mix_info = sf.info(mix_path)
    return mix_info.frames


for split_long, split_short in SPLIT_NAMES.items():
    print('Running {} Set'.format(split_long))
    filelist_path = FILELIST_STUB.format(split_short)
    filelist_df = pd.read_csv(filelist_path)
    utt_ids = list(filelist_df['output_filename'])

    output_dir = os.path.join(OUTPUT_PATH, split_short)
    os.makedirs(output_dir, exist_ok=True)

    nz_sampler = NoiseSampler(NOISE_SPLIT_CSV, RAW_NOISE_16K_PATH, DENOISE_16K_PATH, split=split_long)

    utt_list, noise_param_list, mix_param_list = [], [], []
    for i_utt, utt in enumerate(utt_ids):
        n_speech_samples = get_wsjmix_length(utt, split_short)
        noise_samples, noise_param_dict, mix_param_dict = nz_sampler.sample_utt_noise(n_speech_samples)

        sf.write(os.path.join(output_dir, utt), noise_samples, SAMPLERATE,  subtype='FLOAT')

        utt_list.append(utt)
        noise_param_list.append(noise_param_dict)
        mix_param_list.append(mix_param_dict)

        if (i_utt + 1) % 500 == 0:
            print('Completed {} of {} utterances'.format(i_utt + 1, len(utt_ids)))

    noise_param_df = pd.DataFrame(data=noise_param_list, index=utt_list,
                                  columns=['noise_file', 'start_sample_16k', 'end_sample_16k', 'noise_snr'])
    noise_param_path = os.path.join(METADATA_DIR, 'noise_meta_{}.csv'.format(split_short))
    noise_param_df.to_csv(noise_param_path, index=True, index_label='utterance_id')
    mix_param_df = pd.DataFrame(data=mix_param_list, index=utt_list,
                                columns=['noise_samples_beginning_16k', 'noise_samples_end_16k', 'target_speaker1_snr_db'])
    mix_param_path = os.path.join(METADATA_DIR, 'mix_param_meta_{}.csv'.format(split_short))
    mix_param_df.to_csv(mix_param_path, index=True, index_label='utterance_id')

