import re
from pathlib import Path

import tqdm
import torch
import numpy as np
import pandas as pd
from facenet_pytorch import MTCNN, InceptionResnetV1

from src.loader import Signal, input_face_embeddings, convert_to_spectrogram


class AVDataset(torch.utils.data.Dataset):

    def __init__(self, input_df_path: Path,
                input_audio_size=2):
        """

            Args:
                input_df_path: path for combination dataset
                input_audio_size: total audio/video inputs
        """
        self.input_audio_size = input_audio_size
        self.input_df = pd.read_csv(input_df_path.as_posix())

    def __len__(self):
        return len(self.input_df)

    def __getitem__(self, idx):
        row = self.input_df.iloc[idx, :]
        all_signals = []

        for i in range(self.input_audio_size):
            #get audio, video path from combination dataframe
            video_path = row[i]
            audio_path = row[i+self.input_audio_size]

            #video length is 3-10 seconds, hence, part index can take values 0-2
            re_match = re.search(r'_part\d', audio_path)
            video_length_idx = 0
            if re_match:
                video_length_idx = int(re_match.group(0)[-1])

            signal = Signal(video_path, audio_path, video_start_length=video_length_idx)
            all_signals.append(signal)

        #input audio signal is the last column.
        mixed_signal, is_spec, spec_path = Signal.load_audio(row[-1])

        if not is_spec:
            mixed_signal = convert_to_spectrogram(mixed_signal)

        audio_tensors = []
        video_tensors = []

        for i in range(self.input_audio_size):
            #audio to spectrogram
            if all_signals[i].spec_path.is_file():
                spectrogram =  all_signals[i].get_spec()
            else:
                spectrogram = convert_to_spectrogram(all_signals[i].get_audio())
            #convert to tensor
            audio_tensors.append(torch.from_numpy(spectrogram))

            #check if the embedding is saved
            if all_signals[i].embed_is_saved() and all_signals[i].get_embed() is not None:
                embeddings = torch.from_numpy(all_signals[i].get_embed())
                video_tensors.append(embeddings)
                continue

        # video tensors are expected to be (75,1,1024) (h,w,c)
        # list of video tensors where len(list) == num_person
        # so transpose to be of form video_input = list of video tensors (1024,75,1)
        # we will do
        # for i in range(num_person):
        #   slice out each one , video_input[i] (because this will be of (1024,75,1))

        mixed_signal_tensor = torch.Tensor(mixed_signal)  #shape  (257,298,2)
        mixed_signal_tensor = torch.transpose(mixed_signal_tensor,0,2) #shape (2,298,257)  , therefore , 2 channels , height = 298 , width = 257
        audio_tensors = [i.transpose(0, 2) for i in audio_tensors]
        audio_tensors = torch.stack(audio_tensors)
        audio_tensors = audio_tensors.permute(1, 2, 3, 0)

        return audio_tensors, video_tensors, mixed_signal_tensor

def _check_all_embed_saved(path, num_video):
    files = Path(path).rglob("*")
    files = filter(lambda x: x.contains("npy"), files)

    return len(files == num_video)


def main(args):

    train_dataset = AVDataset(args.train_path, all_embed_saved=False)
    train_df = train_dataset.input_df
    train_video_num = _get_video_num(train_df)

    val_dataset = AVDataset(args.val_path, all_embed_saved=False)
    val_df = val_dataset.input_df
    val_video_num = _get_video_num(val_df)

    total_video_num = train_video_num + val_video_num

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)

    for a, v, m in tqdm.tqdm(train_loader, total=len(train_loader)):
        if _check_all_embed_saved(args.embed_dir, train_video_num):
            print("Saved all training embed")
            break

    for a, v, m in tqdm.tqdm(val_loader, total=len(val_loader)):
        if _check_all_embed_saved(args.embed_dir, total_video_num):
            print("Saved all validation embed")
            break


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--video-dir", default="../../data/train/", type=Path)
    parser.add_argument("--embed-dir", default="../../data/embed/", type=Path)
    parser.add_argument("--train-path", default="../train.csv", type=Path)
    parser.add_argument("--val-path", default="../val.csv", type=Path)

    args = parser.parse_args()

    main(args)
