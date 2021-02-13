import os
import time
import tqdm
import argparse
import subprocess
import pandas as pd
from pathlib import Path
import concurrent.futures

from constants import VIDEO_DIR


def download(link, path, final_name=None):
    command = "youtube-dl {} --no-check-certificate --output {}.mp4 -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4'"
    if os.path.exists(path) and os.path.isfile(path):
        print("File already downloaded")
        return False
    if final_name is not None and os.path.isfile(final_name):
        print("File already cropped")
        return True

    p = subprocess.Popen(
        command.format(link, path),
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).communicate()
    return False


def crop(path, start, end, downloaded_name):
    command = (
        "ffmpeg -y -i {}.mp4 -ss {} -t {} -c:v libx264 -crf 18 -preset veryfast -pix_fmt yuv420p "
        "-c:a aac -b:a 128k -strict experimental -r 25 {}"
    )

    start_minute, start_second = int(start // 60), int(start % 60)
    end_minute, end_second = int(end // 60) - start_minute, int(end % 60) - start_second

    new_filepath = downloaded_name + "_final.mp4"

    if os.path.exists(new_filepath) and os.path.isfile(new_filepath):
        return

    command = command.format(
        downloaded_name,
        f"{start_minute}:{start_second}",
        f"{end_minute}:{end_second}",
        new_filepath,
    )
    subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ).communicate()

    remove_orig_file = f"rm -f {downloaded_name}.mp4"
    subprocess.Popen(
        remove_orig_file, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ).communicate()


def save_video(zargs):
    link, path, start, end, pos_x, pos_y = zargs
    x = int(pos_x * 10000)
    y = int(pos_y * 10000)
    downloaded_name = path.as_posix() + f"_{x}_{y}"
    cropped = download(link, downloaded_name, final_name=downloaded_name + "_final.mp4")
    if not cropped:
        crop(path, start, end, downloaded_name)


def main(args):
    df = pd.read_csv(args.path)
    links = df.iloc[:, 0][args.start : args.end]
    start_times = df.iloc[:, 1][args.start : args.end]
    end_times = df.iloc[:, 2][args.start : args.end]
    pos_x = df.iloc[:, 3][args.start : args.end]
    pos_y = df.iloc[:, 4][args.start : args.end]

    yt_links = ["https://youtube.com/watch\?v\=" + l for l in links]
    paths = [Path(os.path.join(args.vid_dir, f)) for f in links]

    link_path = zip(yt_links, paths, start_times, end_times, pos_x, pos_y)
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as executor:
        results = list(tqdm.tqdm(executor.map(save_video, link_path), total=len(links)))


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="Download parameters")
    parse.add_argument("--jobs", type=int, default=1)
    parse.add_argument("--path", type=str, default="../../data/audio_visual/avspeech_train.csv")
    parse.add_argument("--vid-dir", type=str, default=VIDEO_DIR)
    parse.add_argument("--start", type=int, default=0)
    parse.add_argument("--end", type=int, default=10_000)
    args = parse.parse_args()
    main(args)
