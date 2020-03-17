import os
import glob
import random
import soundfile as sf
from scipy.signal import resample_poly
import argparse


def mix(sources_dir, destination_dir):
    """ Mix 2 sources from a directory and write the mixture in destination"""
    sources = os.listdir(sources_dir)
    L = []
    c = 0
    while len(sources) > 1 and c < 20:
        couple = random.sample(sources, 2)
        if couple[0].split('-')[0] == couple[1].split('-')[0]:
            c += 1
            continue
        else:
            sources.remove(couple[0])
            sources.remove(couple[1])
            L.append(couple)
            c = 0

    for el in L:
        s1, rate_1 = sf.read(os.path.join(sources_dir, el[0]), dtype="float32")
        s2, rate_2 = sf.read(os.path.join(sources_dir, el[1]), dtype="float32")
        mixture = s1 + s2
        sf.write(os.path.join(destination_dir, el[0] + "_" + el[1] + ".flac"),
                 mixture, rate_1)


def generate_sources_mixtures(args):
    in_root = args.in_dir
    out_root = args.out_dir

    os.mkdir(os.path.join(out_root, 'libri2mix'))
    out_root = os.path.join(out_root, 'libri2mix')

    in_root = os.path.join(in_root, 'LibriSpeech')

    dirs = next(os.walk(in_root))[1]

    dirs = sorted(dirs)[:]
    random.seed(123)

    freqs = ['16K', '8K']
    targets = ['cv', 'test', 'train']  # This is sorted

    # Create 8k and 16k directories on root directory
    path_16K = os.path.join(out_root, '16K')
    path_8K = os.path.join(out_root, '8K')

    os.mkdir(os.path.join(out_root, '16K'))
    os.mkdir(os.path.join(out_root, '8K'))

    # Create sources directories
    sources_path_16K = os.path.join(path_16K, 'sources')
    sources_path_8K = os.path.join(path_8K, 'sources')

    os.mkdir(os.path.join(path_16K, 'sources'))
    os.mkdir(os.path.join(path_8K, 'sources'))

    for i, direc in enumerate(dirs):

        path_dir = os.path.join(in_root, direc)

        # Look for .flac files in every subdirectories
        flac_files = os.path.join(path_dir, '**/*.flac')
        flac_list = glob.glob(flac_files, recursive=True)

        target_path_16k = os.path.join(sources_path_16K, targets[i])
        target_path_8k = os.path.join(sources_path_8K, targets[i])
        os.mkdir(target_path_16k)
        os.mkdir(target_path_8k)

        # Create sources of 4 seconds at 8 KHz and 16 KHz
        for file in flac_list:
            file_name = os.path.split(file)[1]
            data, rate = sf.read(file, frames=16000 * 4, dtype='float32')
            if len(data) < 64000:
                pass
            else:
                data_8k = resample_poly(data, 8000, rate)
                sf.write(os.path.join(target_path_16k, file_name), data, rate)
                sf.write(os.path.join(target_path_8k, file_name), data_8k,
                         8000)

    # Create mixtures
    for freq in freqs:
        path = os.path.join(out_root, freq)

        mixtures_path = os.path.join(path, 'mixtures')
        os.mkdir(os.path.join(path, 'mixtures'))

        sources_path = os.path.join(path, 'sources')
        dirs = sorted(os.listdir(sources_path))

        for i, direc in enumerate(dirs):
            path_dir = os.path.join(sources_path, direc)

            target_path = os.path.join(mixtures_path, targets[i])
            os.mkdir(target_path)

            mix(path_dir, target_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, default=None,
                        help='Directory path of LibriSpeech')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Directory path to put output files')
    args = parser.parse_args()
    generate_sources_mixtures(args)
