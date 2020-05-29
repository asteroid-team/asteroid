import re
from pathlib import Path
import torch
from torch.utils import data
import pandas as pd


class AVSpeechDataset(data.Dataset):

    def __init__(self, input_df_path: Path,
                signal_constructor, convert_to_spectrogram,
                input_audio_size=2):
        """

            Args:
                input_df_path: path for combination dataset
                input_audio_size: total audio/video inputs
        """
        self.input_audio_size = input_audio_size
        self.input_df = pd.read_csv(input_df_path.as_posix())
        self.signal_constructor = signal_constructor
        self.convert_to_spectrogram = convert_to_spectrogram

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

            signal = self.signal_constructor(video_path, audio_path, video_start_length=video_length_idx)
            all_signals.append(signal)

        #input audio signal is the last column.
        mixed_signal, is_spec, spec_path = self.signal_constructor.load_audio(row[-1])

        if not is_spec:
            mixed_signal = self.convert_to_spectrogram(mixed_signal)

        audio_tensors = []
        video_tensors = []

        for i in range(self.input_audio_size):
            #audio to spectrogram
            if all_signals[i].spec_path.is_file():
                spectrogram =  all_signals[i].get_spec()
            else:
                spectrogram = self.convert_to_spectrogram(all_signals[i].get_audio())
            #convert to tensor
            audio_tensors.append(torch.from_numpy(spectrogram))

            #check if the embedding is saved
            if all_signals[i].embed_is_saved() and all_signals[i].get_embed() is not None:
                embeddings = torch.from_numpy(all_signals[i].get_embed())
                video_tensors.append(embeddings)
                continue

        mixed_signal_tensor = torch.Tensor(mixed_signal)  #shape  (257,298,2)
        mixed_signal_tensor = torch.transpose(mixed_signal_tensor,0,2) #shape (2,298,257)  , therefore , 2 channels , height = 298 , width = 257
        audio_tensors = [i.transpose(0, 2) for i in audio_tensors]
        audio_tensors = torch.stack(audio_tensors)
        audio_tensors = audio_tensors.permute(1, 2, 3, 0)

        return audio_tensors, video_tensors, mixed_signal_tensor

