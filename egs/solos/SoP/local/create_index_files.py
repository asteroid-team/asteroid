import os
import glob
import argparse
import random
import fnmatch


def find_recursive(root_dir, ext='.mp3'):
    files = []
    for root, dirnames, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, '*' + ext):
            files.append(os.path.join(root, filename))
    return files


def create_files(src, dst, fps, ratio):
    src_audio = os.path.join(src, 'audio')
    src_frames = os.path.join(src, 'frames')
    # find all audio/frames pairs
    infos = []
    audio_files = find_recursive(src_audio, ext='.mp3')
    for audio_path in audio_files:
        frame_path = audio_path.replace(src_audio, src_frames) \
            .replace('.mp3', '')
        frame_files = glob.glob(frame_path + '/*.png')
        if len(frame_files) > fps * 20:
            infos.append(','.join([audio_path, frame_path, str(len(frame_files))]))
    print('{} audio/frames pairs found.'.format(len(infos)))

    # split train/val
    n_train = int(len(infos) * 0.8)
    random.shuffle(infos)
    trainset = infos[0:n_train]
    valset = infos[n_train:]
    for name, subset in zip(['train', 'val'], [trainset, valset]):
        filename = '{}.csv'.format(os.path.join(dst, name))
        with open(filename, 'w') as f:
            for item in subset:
                f.write(item + '\n')
        print('{} items saved to {}.'.format(len(subset), filename))

    print('Done!')
