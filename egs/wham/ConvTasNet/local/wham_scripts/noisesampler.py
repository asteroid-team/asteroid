import os
import pandas as pd
import numpy as np
import soundfile as sf
from constants import NUM_BANDS, SNR_THRESH, PRE_NOISE_SAMPLES, MAX_SNR_DB, MIN_SNR_DB


class NoiseSampler:
    def __init__(self, csv_filelist, audio_root, rx_audio_root, split='Train'):
        self.audio_root = audio_root
        self.rx_audio_root = rx_audio_root
        df_file = pd.read_csv(csv_filelist)
        df_split = df_file[df_file['Split'] == split]
        self.band_list = [None] * NUM_BANDS
        for i_band in range(NUM_BANDS):
            files = list(df_split[df_split['Noise Band'] == i_band]['Filename'])
            weights = np.array([self._get_minutes(f) for f in files])
            weights /= np.sum(weights)
            self.band_list[i_band] = {'files': files, 'weights': weights}

    def _get_path(self, fname, read_denoised=False):
        if read_denoised:
            return os.path.join(self.rx_audio_root, fname)
        return os.path.join(self.audio_root, fname)

    def _get_minutes(self, fname):
        wav_path = self._get_path(fname)
        return sf.info(wav_path).duration / 60

    def _get_num_frames(self, fname):
        wav_path = self._get_path(fname)
        return sf.info(wav_path).frames

    def _sample_file(self):
        noise_band = np.random.randint(0, NUM_BANDS)
        cur_band = self.band_list[noise_band]
        return np.random.choice(cur_band['files'], p=cur_band['weights'])

    def _sample_frames(self, fname, len_samples):
        wav_frames = self._get_num_frames(fname)
        start_frame = np.random.randint(0, wav_frames-len_samples)
        end_frame = start_frame + len_samples
        return start_frame, end_frame

    def _check_snr(self, fname, start_sample, end_sample):
        noise, _ = sf.read(self._get_path(fname), start=start_sample, stop=end_sample)
        denoise, _ = sf.read(self._get_path(fname, read_denoised=True),
                             start=start_sample, stop=end_sample)
        residual = noise - denoise
        sig_avg = np.mean(np.square(denoise))
        noise_avg = np.mean(np.square(residual))
        snr = 10 * np.log10(sig_avg / (noise_avg + 1e-10) + 1e-10)
        return snr, noise


    def _sample_noise(self, utt_len_samples):
        snr = np.inf
        while snr > SNR_THRESH:
            f = self._sample_file()
            start_sample, end_sample = self._sample_frames(f, utt_len_samples)
            snr, samples = self._check_snr(f, start_sample, end_sample)
        return samples, f, start_sample, end_sample, snr


    def sample_utt_noise(self, n_signal_samples):
        noise_beginning, noise_end = np.random.randint(0, PRE_NOISE_SAMPLES, size=2)
        target_snr = np.random.uniform(MIN_SNR_DB, MAX_SNR_DB)
        noise_samples_to_read = n_signal_samples + noise_beginning + noise_end
        noise_samples, f_nz, start_samp_nz, end_samp_nz, est_nz_snr = \
            self._sample_noise(noise_samples_to_read)

        noise_meta = { 'noise_file': f_nz,
                       'start_sample_16k': start_samp_nz,
                       'end_sample_16k': end_samp_nz,
                       'noise_snr': est_nz_snr }

        mix_meta = { 'noise_samples_beginning_16k': noise_beginning,
                     'noise_sample_end_16k': noise_end,
                     'target_speaker1_snr_db': target_snr}

        return noise_samples, noise_meta, mix_meta
