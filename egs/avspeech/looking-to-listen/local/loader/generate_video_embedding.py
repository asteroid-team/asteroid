import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from asteroid.data.avspeech_dataset import get_frames
from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face

from frames import input_face_embeddings
from constants import VIDEO_DIR, EMBED_DIR

FRAMES = 75


def _get_video(df):
    video_columns = [i for i in list(df) if "video" in i]

    video_paths = df[video_columns].values.reshape(-1).tolist()
    video_paths = sorted(set(video_paths), key=video_paths.index)

    return video_paths


def store_corrupt(path):
    with open(args.corrupt_file, "a") as f:
        f.write(path.as_posix() + "\n")


def cache_embed(path, mtcnn, resnet, args):
    orig_path = path
    if not path.is_file():
        path = "../.." / path
    video_file_name = path.stem.split("_")

    if len(video_file_name) < 3:
        store_corrupt(orig_path)
        return

    try:
        pos_x, pos_y = (
            int(video_file_name[-3]) / 10000,
            int(video_file_name[-2]) / 10000,
        )
    except ValueError as e:
        print(str(e))
        store_corrupt(orig_path)
        return

    video_buffer = get_frames(cv2.VideoCapture(path.as_posix()))
    total_frames = video_buffer.shape[0]

    video_parts = total_frames // FRAMES  # (25fps * 3)

    embeddings = []
    for part in range(video_parts):
        frame_name = path.stem + f"_part{part}"
        embed_path = Path(args.embed_dir, frame_name + ".npy")
        if embed_path.is_file():
            continue
        raw_frames = video_buffer[part * FRAMES : (part + 1) * FRAMES]

        embed = input_face_embeddings(
            raw_frames,
            is_path=False,
            mtcnn=mtcnn,
            resnet=resnet,
            face_embed_cuda=args.cuda,
            use_half=args.use_half,
            coord=[pos_x, pos_y],
        )

        if embed is None:
            store_corrupt(orig_path)
            print("Corrupt", path)
            return

        embeddings.append((embed, embed_path))

    # save if all parts are not corrupted
    for embed, embed_path in embeddings:
        np.save(embed_path, embed.cpu().numpy())


def main(args):
    train_df = pd.read_csv(args.train_path)
    val_df = pd.read_csv(args.val_path)

    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    mtcnn = MTCNN(keep_all=True).eval().to(device)
    mtcnn.device = device

    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    video_paths = _get_video(train_df)
    video_paths += _get_video(val_df)

    print(f"Total embeddings: {len(video_paths)}")
    for path in tqdm(video_paths, total=len(video_paths)):
        cache_embed(Path(path), mtcnn, resnet, args)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--video-dir", default=Path(VIDEO_DIR), type=Path)
    parser.add_argument("--embed-dir", default=Path(EMBED_DIR), type=Path)
    parser.add_argument("--train-path", default=Path("../../data/train.csv"), type=Path)
    parser.add_argument("--val-path", default=Path("../../data/val.csv"), type=Path)
    parser.add_argument("--cuda", dest="cuda", action="store_true")
    parser.add_argument("--use-half", dest="use_half", action="store_true")
    parser.add_argument(
        "--corrupt-file", default=Path("../../data/corrupt_frames_list.txt"), type=Path
    )

    args = parser.parse_args()

    main(args)
