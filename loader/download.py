import os
import time
import argparse
import subprocess
import pandas as pd
from pathlib import Path
import concurrent.futures


def download(link, cwd="../../data/train"):
    command = "youtube-dl {} --output {}.mp4 -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4'"
    path = os.path.join(cwd, link + ".mp4")
    if os.path.exists(path) and os.path.isfile(path):
        return
    p = subprocess.Popen(command.format(link, link[-11:]), shell=True, stdout=subprocess.PIPE, cwd=cwd).communicate()


def crop(path, start, end):
    command = "ffmpeg -y -i {} -ss {} -t {} -c:v libx264 -crf 18 -preset veryfast -pix_fmt yuv420p -c:a aac -b:a 128k -strict experimental -r 25 {}" 

    start_minute, start_second = int(start // 60), int(start % 60)
    end_minute, end_second = int(end // 60) - start_minute, int(end % 60) - start_second

    parent = path.parents[0]
    name = path.stem
    new_filepath = os.path.join(parent, name + "_cropped.mp4")

    command = command.format(path.as_posix(), f"{start_minute}:{start_second}", f"{end_minute}:{end_second}", new_filepath)
    print(command)

    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).communicate()


def save_video(link, path, start, end):
    download(link)
    crop(path, start, end)


def main(args):
    df = pd.read_csv(args.path)
    links = df.iloc[:, 0][:2]
    start_times = df.iloc[:, 1][:2]
    end_times = df.iloc[:, 2][:2]
    
    yt_links = ["https://youtube.com/watch\?v\="+l for l in links]
    paths = [Path(os.path.join(args.vid_dir, f + ".mp4")) for f in links]

    link_path = zip(yt_links, paths, start_times, end_times)
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as executor:
        futures = [executor.submit(save_video, l, p, s, e) for l, p, s, e in link_path]

        for f in concurrent.futures.as_completed(futures):
            pass


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="Download parameters")
    parse.add_argument("--jobs", type=int, default=2)
    parse.add_argument("--path", type=str, default="../../data/audio_visual/avspeech_train.csv")
    parse.add_argument("--vid-dir", type=str, default="../../data/train/")
    args = parse.parse_args()
    main(args)
    
