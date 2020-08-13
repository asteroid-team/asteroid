import re
import cv2
import librosa
import numpy as np
from pathlib import Path
import torch
from torch.utils import data
from torch.nn import functional as F
import pandas as pd
from typing import Union
from asteroid.filterbanks import Encoder, Decoder, STFTFB

EPS = 1e-8


def get_frames(video):
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buffer_video = np.empty((frame_count, frame_height, frame_width, 3), np.dtype("uint8"))

    frame = 0
    ret = True

    while frame < frame_count and ret:
        ret, f = video.read()
        buffer_video[frame] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)

        frame += 1
    video.release()
    return buffer_video


class Signal:
    """This class holds the video frames and the audio signal.

        Args:
            video_path (str,Path): Path to video (mp4).
            audio_path (str,Path): Path to audio (wav).
            embed_dir (str,Path): Path to directory that stores embeddings.
            sr (int): sampling rate of audio.
            video_start_length: video part no. [1]
            fps (int): fps of video.
            signal_len (int): length of the signal

        .. note:: each video consists of multiple parts which consists of fps*signal_len frames.
    """

    def __init__(
        self,
        video_path: Union[str, Path],
        audio_path: Union[str, Path],
        embed_dir: Union[str, Path],
        sr=16_000,
        video_start_length=0,
        fps=25,
        signal_len=3,
    ):
        if isinstance(video_path, str):
            video_path = Path(video_path)
        if isinstance(audio_path, str):
            audio_path = Path(audio_path)
        if isinstance(embed_dir, str):
            embed_dir = Path(embed_dir)

        self.video_path = video_path
        self.audio_path = audio_path
        self.video_start_length = video_start_length

        self.embed_path = None
        self.embed = None
        self.embed_dir = embed_dir

        self.fps = fps
        self.signal_len = signal_len
        self.sr = sr

        self._load(sr=sr)
        self._check_video_embed()

    def _load(self, sr: int):
        self.audio, _ = librosa.load(self.audio_path.as_posix(), sr=sr)
        self.video = cv2.VideoCapture(self.video_path.as_posix())

    def _check_video_embed(self, embed_ext=".npy"):
        # convert mp4 location to embedding...
        video_name_stem = self.video_path.stem

        embed_dir = self.embed_dir
        if not embed_dir.is_dir():
            # check embed_dir="../../dir" or embed_dir="dir"
            embed_dir = Path(*embed_dir.parts[2:])

        self.embed_path = Path(
            embed_dir, f"{video_name_stem}_part{self.video_start_length}{embed_ext}"
        )
        if self.embed_path.is_file():
            self.embed = np.load(self.embed_path.as_posix())
        else:
            raise ValueError(
                f"Embeddings not found in {self.embed_dir} for {self.video_path} "
                f"for part: {self.video_start_length}"
            )

    def get_embed(self):
        return self.embed

    def get_audio(self):
        return self.audio


class AVSpeechDataset(data.Dataset):
    """Audio Visual Speech Separation dataset as described in [1].

        Args:
            input_df_path (str,Path): path for combination dataset.
            embed_dir (str,Path): path where embeddings are stored.
            n_src (int): number of sources.

        References:
            [1]: 'Looking to Listen at the Cocktail Party:
            A Speaker-Independent Audio-Visual Model for Speech Separation' Ephrat et. al
            https://arxiv.org/abs/1804.03619
    """

    dataset_name = "AVSpeech"

    def __init__(self, input_df_path: Union[str, Path], embed_dir: Union[str, Path], n_src=2):
        if isinstance(input_df_path, str):
            input_df_path = Path(input_df_path)
        if isinstance(embed_dir, str):
            embed_dir = Path(embed_dir)

        self.n_src = n_src
        self.embed_dir = embed_dir
        self.input_df = pd.read_csv(input_df_path.as_posix())
        self.stft_encoder = Encoder(STFTFB(n_filters=512, kernel_size=400, stride=160))

    @staticmethod
    def encode(x: np.ndarray, p=0.3, stft_encoder=None):
        if stft_encoder is None:
            stft_encoder = Encoder(STFTFB(n_filters=512, kernel_size=400, stride=160))

        x = torch.from_numpy(x).float()

        # time domain to time-frequency representation
        tf_rep = stft_encoder(x).squeeze(0) + EPS
        # power law on complex numbers
        tf_rep = (torch.abs(tf_rep) ** p) * torch.sign(tf_rep)
        return tf_rep

    @staticmethod
    def decode(tf_rep: np.ndarray, p=0.3, stft_decoder=None, final_len=48000):
        if stft_decoder is None:
            stft_decoder = Decoder(STFTFB(n_filters=512, kernel_size=400, stride=160))

        tf_rep = torch.from_numpy(tf_rep).float()

        # power law on complex numbers
        tf_rep = (torch.abs(tf_rep) ** (1 / p)) * torch.sign(tf_rep)
        # time domain to time-frequency representation
        x = stft_decoder(tf_rep)

        length = len(x)
        if length != final_len:
            x = F.pad(x, [0, final_len - length])
        return x

    def __len__(self):
        return len(self.input_df)

    def __getitem__(self, idx):
        row = self.input_df.iloc[idx, :]
        all_signals = []

        for i in range(self.n_src):
            # get audio, video path from combination dataframe
            video_path = row.loc[f"video_{i+1}"]
            audio_path = row.loc[f"audio_{i+1}"]

            # video length is 3-10 seconds, hence, part index can take values 0-2
            re_match = re.search(r"_part\d", audio_path)
            video_length_idx = 0
            if re_match:
                video_length_idx = int(re_match.group(0)[-1])

            signal = Signal(
                video_path, audio_path, self.embed_dir, video_start_length=video_length_idx,
            )
            all_signals.append(signal)

        # input audio signal is the last column.
        mixed_signal, _ = librosa.load(row.loc["mixed_audio"], sr=16_000)
        mixed_signal_tensor = self.encode(mixed_signal, stft_encoder=self.stft_encoder)

        audio_tensors = []
        video_tensors = []

        for i in range(self.n_src):
            # audio to spectrogram
            spectrogram = self.encode(all_signals[i].get_audio(), stft_encoder=self.stft_encoder)
            audio_tensors.append(spectrogram)

            # get embed
            embeddings = torch.from_numpy(all_signals[i].get_embed())
            video_tensors.append(embeddings)

        audio_tensors = torch.stack(audio_tensors)

        return audio_tensors, video_tensors, mixed_signal_tensor
