import os
import torch
import pandas as pd
from data import Signal
from pathlib import Path


class AVDataset(torch.utils.data.Dataset):
    

    def __init__(self, dataset_df_path: Path, video_base_dir: Path, input_df_path: Path, input_audio_size=2):
        self.input_audio_size = input_audio_size

        self.dataset_df = pd.read_csv(dataset_df_path.as_posix())
        self.file_names = self.dataset_df.iloc[:, 0]
        self.file_names = [os.path.join(video_base_dir.as_posix(), f + "_cropped.mp4") 
                        for f in self.file_names]
        self.start_times = self.dataset_df.iloc[:, 1]
        self.end_times = self.dataset_df.iloc[:, 2]

        self.face_x = self.dataset_df.iloc[:, 3]
        self.face_y = self.dataset_df.iloc[:, 4]

        self.input_df = pd.read_csv(input_df_path.as_posix())

    def __len__(self):
        return len(self.input_df)

    def __getitem__(self, idx):
        row = self.input_df.iloc[idx, :]
        all_signals = []
        
        for i in range(self.input_audio_size):
            video_path = row[i]
            audio_path = row[i+self.input_audio_size]

            signal = Signal(video_path, audio_path)
            all_signals.append(signal)
        mixed_signal = Signal.load_audio(row[-1])

        audio_tensors = []
        video_tensors = []

        for i in range(self.input_audio_size):
            audio_tensors.append(torch.from_numpy(all_signals[i].get_audio()))
            video_tensors.append(torch.from_numpy(all_signals[i].get_video()))

        mixed_signal_tensor = torch.Tensor(mixed_signal)

        return audio_tensors, video_tensors, mixed_signal_tensor


if __name__ == "__main__":
    dataset = AVDataset(Path("../../data/audio_visual/avspeech_train.csv"),
                      Path("../../data/train/"),
                      Path("temp.csv"))
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    for a, v, m in loader:
        print(len(a), len(v), a[0].shape, v[0].shape, m.shape)
