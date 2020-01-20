import os
import time
import argparse
import subprocess
import pandas as pd
from pathlib import Path
import concurrent.futures


def download(link, path):
    command = "youtube-dl {} --output {}.mp4 -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4'"
    if os.path.exists(path) and os.path.isfile(path):
        return
    print(command.format(link, path))
    p = subprocess.Popen(command.format(link, path), shell=True, stdout=subprocess.PIPE, cwd=args.vid_dir).communicate()


def crop(path, start, end, resolution, downloaded_name):
    command = "ffmpeg -y -i {}.mp4 -ss {} -t {} -c:v libx264 -crf 18 -preset veryfast -pix_fmt yuv420p -c:a aac -b:a 128k -strict experimental -r 25 {}" 

    start_minute, start_second = int(start // 60), int(start % 60)
    end_minute, end_second = int(end // 60) - start_minute, int(end % 60) - start_second

    parent = path.parents[0]
    new_filepath = downloaded_name + "_cropped.mp4"
    low_res_filepath = downloaded_name + "_final.mp4"
    print(new_filepath, low_res_filepath)
    if os.path.exists(low_res_filepath) and os.path.isfile(low_res_filepath):
        return

    command = command.format(downloaded_name, f"{start_minute}:{start_second}", f"{end_minute}:{end_second}", new_filepath)
    print(command)

    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).communicate()

    downsample = f"avconv -i {new_filepath} -s {resolution} {low_res_filepath} -y"
    p = subprocess.Popen(downsample, shell=True, stdout=subprocess.PIPE).communicate()


def save_video(link, path, start, end, resolution, pos_x, pos_y):
    x = int(pos_x*10000)
    y = int(pos_y*10000)
    downloaded_name = path.as_posix() + f"_{x}_{y}"
    download(link, downloaded_name)
    crop(path, start, end, resolution, downloaded_name)

def main(args):
    df = pd.read_csv(args.path)
    links = df.iloc[:, 0][500:700]
    start_times = df.iloc[:, 1][500:700]
    end_times = df.iloc[:, 2][500:700]
    pos_x = df.iloc[:, 3][500:700]
    pos_y = df.iloc[:, 4][500:700]
    
    yt_links = ["https://youtube.com/watch\?v\="+l for l in links]
    paths = [Path(os.path.join(args.vid_dir, f)) for f in links]

    link_path = zip(yt_links, paths, start_times, end_times, pos_x, pos_y)
    [save_video(l, p, s, e, args.resolution, x, y) for l, p, s, e, x, y, in link_path]
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as executor:
        futures = [executor.submit(save_video, l, p, s, e, args.resolution, x, y) for l, p, s, e, x, y in link_path]

        for f in concurrent.futures.as_completed(futures):
            pass


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="Download parameters")
    parse.add_argument("--jobs", type=int, default=1)
    parse.add_argument("--path", type=str, default="../../data/audio_visual/avspeech_train.csv")
    parse.add_argument("--vid-dir", type=str, default="../../data/train/")
    parse.add_argument("--resolution", type=str, default="320x200")
    args = parse.parse_args()
    main(args)
    
