import os
import time
import argparse
import subprocess
import pandas as pd
from pathlib import Path
import concurrent.futures


def extract(path):
    name = path.stem

    dir_name = path.parents[0]
    audio_dir = args.aud_dir
    audio_path = os.path.join(dir_name, audio_dir, name)

    command = "ffmpeg -i {} -f mp3 -ab 192000 -vn {}.mp3"
    command = command.format(path.as_posix(), audio_path)

    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).communicate()


def main(args):
    file_names = [Path(os.path.join(args.vid_dir, i)) for i in os.listdir(args.path) if i.endswith("_cropped.mp4")]

    with concurrent.futures.ThreadPoolExecutor(args.jobs) as executor:
        futures = [executor.submit(extract, f) for f in file_names]

        for f in concurrent.futures.as_completed(futures):
            pass


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="Download parameters")
    parse.add_argument("--jobs", type=int, default=2)
    parse.add_argument("--path", type=str, default="../../data/train/")
    parse.add_argument("--aud-dir", type=str, default="../../data/train/audio")
    parse.add_argument("--vid-dir", type=str, default="../../data/train/")
    args = parse.parse_args()
    main(args)
