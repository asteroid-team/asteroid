import os
import cv2
import time
import argparse
import subprocess
import pandas as pd
from pathlib import Path
import concurrent.futures
from tqdm import tqdm

from constants import AUDIO_DIR, VIDEO_DIR


def extract(path):
    name = path.stem

    dir_name = path.parents[0]
    audio_dir = args.aud_dir
    audio_path = os.path.join(audio_dir, name)
    video = cv2.VideoCapture(path.as_posix())
    length_orig_video = video.get(cv2.CAP_PROP_FRAME_COUNT)
    # already pre-processed at 25 fps for 3 or more seconds
    length = int(length_orig_video) // 25 // 3
    for i in range(length):
        t = i * 3
        command = (
            f"ffmpeg -y -i {path.as_posix()} -f {args.audio_extension} -ab 64000 "
            f"-vn -ar {args.sampling_rate} -ac {args.audio_channel} - | sox -t "
            f"{args.audio_extension} - -r 16000 -c 1 -b 8 "
            f"{audio_path}_part{i}.{args.audio_extension} trim {t} 00:{args.duration:02d}"
        )

        subprocess.Popen(
            command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ).communicate()


def main(args):
    file_names = [
        Path(os.path.join(args.vid_dir, i))
        for i in os.listdir(args.vid_dir)
        if i.endswith("_final.mp4")
    ]

    with concurrent.futures.ThreadPoolExecutor(args.jobs) as executor:
        results = list(tqdm(executor.map(extract, file_names), total=len(file_names)))


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="Extract parameters")
    parse.add_argument("--jobs", type=int, default=2)
    parse.add_argument("--aud-dir", type=str, default=AUDIO_DIR)
    parse.add_argument("--vid-dir", type=str, default=VIDEO_DIR)
    parse.add_argument("--sampling-rate", type=int, default=16_000)
    parse.add_argument("--audio-channel", type=int, default=2)
    parse.add_argument("--audio-extension", type=str, default="wav")
    parse.add_argument("--duration", type=int, default=3)
    args = parse.parse_args()
    main(args)
