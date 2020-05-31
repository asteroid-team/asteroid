import re
import os
import cv2
import librosa
import numpy as np
from pathlib import Path
import torch
from torch.utils import data
from torch.nn import functional as F
import pandas as pd

from typing import Callable, Tuple, List

from asteroid.filterbanks import (Encoder, Decoder,
                                  STFTFB)

def get_frames(video):
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buffer_video = np.empty((frame_count, frame_height, frame_width, 3), np.dtype('uint8'))

    frame = 0
    ret = True

    while (frame < frame_count  and ret):
        ret, f = video.read()
        buffer_video[frame] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)

        frame += 1
    video.release()
    return buffer_video

class Signal:
    '''
        This class holds the video frames and the audio signal.
    '''

    def __init__(self, video_path: str, audio_path: str, embed_dir: Path,
                 audio_ext=".mp3", sr=16_000, video_start_length=0,
                 load_spec=True):
        self.video_path = Path(video_path)
        self.audio_path = Path(audio_path)
        self.video_start_length = video_start_length

        self.embed_path = None
        self.embed_saved = False
        self.embed = None
        self.embed_dir = embed_dir

        self.load_spec = load_spec
        self._is_spec = False

        self.spec_path = Path(*self.audio_path.parts[:-2], "spec", self.audio_path.stem + ".npy")

        self._load(sr=sr)
        self._check_video_embed()
        self._convert_video()


    def _load(self, sr: int):
        self.audio = None
        if self.load_spec and self.spec_path.is_file():
            self.audio = np.load(self.spec_path)
            self._is_spec = True
        if self.audio is None:
            self.audio, sr = librosa.load(self.audio_path.as_posix(), sr=sr)
            self._is_spec = False
        self.video = cv2.VideoCapture(self.video_path.as_posix())

    def augment_audio(self, augmenter: Callable, *args, **kwargs):
        '''
            Change the audio via the augmenter method.
        '''
        self.audio = augmenter(self.audio, *args, **kwargs)

    def augment_video(self, augmenter: Callable, *args, **kwargs):
        '''
            Change the video via the augmenter method.
        '''
        self.video = augmenter(self.video, *args, **kwargs)

    def _convert_video(self):
        if self.embed_saved:
            return
        self.buffer_video = get_frames(self.video)

    def _check_video_embed(self, embed_ext=".npy"):
        video_name_stem = self.video_path.stem

        embed_dir = self.embed_dir
        if not embed_dir.is_dir():
            embed_dir = Path(*embed_dir.parts[2:])

        self.embed_path = Path(embed_dir, video_name_stem + f"_part{self.video_start_length}" + embed_ext)
        if self.embed_path.is_file():
            self.embed_saved = True
            self.embed = np.load(self.embed_path.as_posix())

    def embed_is_saved(self):
        return self.embed_saved

    def get_embed(self):
        return self.embed

    def get_video(self):
        #retrieve slice of video, if video > 75 frames
        buffer_video = self.buffer_video[self.video_start_length*75: (self.video_start_length+1)*75]
        return buffer_video

    def get_audio(self):
        return self.audio

    def get_spec(self):
            return np.load(self.spec_path)

    def is_spec(self):
        return self._is_spec

    @staticmethod
    def load_audio(audio_path: str, sr=16_000, load_spec=False):
        audio_path = Path(audio_path)
        spec_exists = False
        spec_path = Path(*audio_path.parts[:-2], "spec", audio_path.stem + ".npy")
        if load_spec and spec_path.is_file():
            spec_exists = True
            audio = np.load(spec_path)
        else:
            audio = librosa.load(str(audio_path), sr=sr)[0]
        return audio, spec_exists, spec_path


class AVSpeechDataset(data.Dataset):

    def __init__(self, input_df_path: Path,
                 embed_dir: Path,
                 input_audio_size=2):
        """

            Args:
                input_df_path: path for combination dataset
                input_audio_size: total audio/video inputs
        """
        self.input_audio_size = input_audio_size
        self.embed_dir = embed_dir
        self.input_df = pd.read_csv(input_df_path.as_posix())

    @staticmethod
    def encode(x: np.ndarray, p=0.3, stft_encoder=None):
        if stft_encoder is None:
            stft_encoder = Encoder(STFTFB(n_filters=512, kernel_size=400,
                                          stride=160))

        x = torch.from_numpy(x).float()

        # time domain to time-frequency representation
        tf_rep = stft_encoder(x).squeeze(0).unsqueeze(2)
        # power law on complex numbers
        tf_rep = (torch.abs(tf_rep) ** p) * torch.sign(tf_rep)
        # (514, 298) -> (257, 298, 2)
        tf_rep = torch.cat((tf_rep[:257], tf_rep[257:]), axis=2)
        return tf_rep

    @staticmethod
    def decode(tf_rep: np.ndarray, p=0.3, stft_decoder=None, final_len=48000):
        if stft_decoder is None:
            stft_decoder = Decoder(STFTFB(n_filters=512, kernel_size=400,
                                          stride=160))

        tf_rep = torch.from_numpy(tf_rep).float()

        # (257, 298, 2) -> (514, 298)
        tf_rep = torch.cat((tf_rep[..., 0], tf_rep[..., 1]), axis=0)
        # power law on complex numbers
        tf_rep = (torch.abs(tf_rep) ** (1/p)) * torch.sign(tf_rep)
        # time domain to time-frequency representation
        x = stft_decoder(tf_rep)

        length = len(x)
        if length != final_len:
            x = F.pad(x, [0, final_len-length])
        return x

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

            signal = Signal(video_path, audio_path, self.embed_dir, video_start_length=video_length_idx)
            all_signals.append(signal)

        #input audio signal is the last column.
        mixed_signal, is_spec, spec_path = Signal.load_audio(row[-1])

        if not is_spec:
            mixed_signal = self.encode(mixed_signal)

        audio_tensors = []
        video_tensors = []

        for i in range(self.input_audio_size):
            #audio to spectrogram
            if all_signals[i].spec_path.is_file():
                spectrogram =  all_signals[i].get_spec()
            else:
                spectrogram = self.encode(all_signals[i].get_audio())
            #convert to tensor
            audio_tensors.append(spectrogram)

            #check if the embedding is saved
            if all_signals[i].embed_is_saved() and all_signals[i].get_embed() is not None:
                embeddings = torch.from_numpy(all_signals[i].get_embed())
                video_tensors.append(embeddings)
                continue

        mixed_signal_tensor = torch.transpose(mixed_signal,0,2) #shape (2,298,257)  , therefore , 2 channels , height = 298 , width = 257
        audio_tensors = [i.transpose(0, 2) for i in audio_tensors]
        audio_tensors = torch.stack(audio_tensors)
        audio_tensors = audio_tensors.permute(1, 2, 3, 0)

        return audio_tensors, video_tensors, mixed_signal_tensor

