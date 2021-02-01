import argparse
from create_wham_from_scratch import create_wham


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str,
                        help='Output directory for writing wsj0-2mix 8 k Hz and 16 kHz datasets.')
    parser.add_argument('--wsjmix-dir-16k', type=str,
                        help='Folder containing original wsj0-2mix 2speakers 16 kHz dataset. Input argument from \
                                 create_wav_2speakers.m Matlab script')
    parser.add_argument('--wsjmix-dir-8k', type=str,
                        help='Folder containing original wsj0-2mix 2speakers 8 kHz dataset. Input argument from \
                                                         create_wav_2speakers.m Matlab script')
    parser.add_argument('--wham-noise-root', type=str,
                        help='Path to the downloaded and unzipped wham folder containing metadata/')
    args = parser.parse_args()
    create_wham(None, args.wham_noise_root, args.output_dir, args.wsjmix_dir_16k, args.wsjmix_dir_8k)