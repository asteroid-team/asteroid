import os
import torch
import pandas as pd
from data import Signal
from pathlib import Path
from audio_feature_generator import convert_to_spectrogram

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
            spectrogram = convert_to_spectrogram(all_signals[i].get_audio())
            audio_tensors.append(torch.from_numpy(spectrogram))

            #TODO: Convert raw video to face embedding
            video_tensors.append(torch.from_numpy(all_signals[i].get_video()))

        # video tensors are expected to be (75,1,1024) (h,w,c)
        # list of video tensors where len(list) == num_person
        # so transpose to be of form video_input = list of video tensors (1024,75,1)
        # we will do
        # for i in range(num_person):
        #   slice out each one , video_input[i] (because this will be of (1024,75,1))

        mixed_signal_tensor = torch.Tensor(convert_to_spectrogram(mixed_signal))  #shape  (257,298,2)
        mixed_signal_tensor = torch.transpose(mixed_signal_tensor,0,2) #shape (2,298,257)  , therefore , 2 channels , height = 298 , width = 257	

        return audio_tensors, video_tensors, mixed_signal_tensor


if __name__ == "__main__":
    dataset = AVDataset(Path("../../data/audio_visual/avspeech_train.csv"),
                      Path("../../data/train/"),
                      Path("temp.csv"))
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    for a, v, m in loader:
        print(len(a), len(v), a[0].shape, v[0].shape, m.shape)
