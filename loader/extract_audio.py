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

    command = f"ffmpeg -y -i {path.as_posix()} -f {args.audio_extension} -ab 64000 -vn -ar {args.sampling_rate} -ac {args.audio_channel} - | sox -t {args.audio_extension} - -r 16000 -c 1 -b 8 {audio_path}.{args.audio_extension} trim 0 00:{args.duration:02d}"

    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).communicate()


def main(args):
    file_names = [Path(os.path.join(args.vid_dir, i)) for i in os.listdir(args.path) if i.endswith("_cropped.mp4")]
    extract(file_names[0])

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
    parse.add_argument("--sampling-rate", type=int, default=16_000)
    parse.add_argument("--audio-channel", type=int, default=2)
    parse.add_argument("--audio-extension", type=str, default="wav")
    parse.add_argument("--duration", type=int, default=3)
    args = parse.parse_args()
    main(args)
