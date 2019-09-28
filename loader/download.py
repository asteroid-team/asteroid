import os
import time
import argparse
import subprocess
import pandas as pd
import concurrent.futures


def download(link, cwd="../../data/train"):
    command = "youtube-dl {} -f 160 --output {}.mp4"
    p = subprocess.Popen(command.format(link, link[-11:]), shell=True, stdout=subprocess.PIPE, cwd=cwd).communicate()


def crop(path, start, end):
    command = "echo yes | medipack trim {} -s {} -e {} -o {}"

    start_minute, start_second = int(start // 60), int(start % 60)
    end_minute, end_second = int(end // 60), int(end % 60)

    command = command.format(path + ".mp4", f"{start_minute}:{start_second}", f"{end_minute}:{end_second}", path + "_cropped.mp4")

    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).communicate()


def save_video(link, path, start, end):
    download(link)
    crop(path, start, end)


def main(args):
    train_df = pd.read_csv(args.train_path)
    links = train_df.iloc[:, 0][:2]
    start_times = train_df.iloc[:, 1][:2]
    end_times = train_df.iloc[:, 2][:2]
    
    yt_links = ["https://youtube.com/watch\?v\="+l for l in links]
    paths = [os.path.join(args.vid_dir, f) for f in links]

    link_path = zip(yt_links, paths, start_times, end_times)
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as executor:
        futures = [executor.submit(save_video, l, p, s, e) for l, p, s, e in link_path]

        for f in concurrent.futures.as_completed(futures):
            pass


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="Download parameters")
    parse.add_argument("--jobs", type=int, default=2)
    parse.add_argument("--train-path", type=str, default="../../data/avspeech_train.csv")
    parse.add_argument("--vid-dir", type=str, default="../../data/train")
    args = parse.parse_args()
    main(args)
    
