import os
import cv2
import librosa
import numpy as np
from pathlib import Path

from typing import Callable, Tuple, List

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

    def __init__(self, video_path: str, audio_path: str, audio_ext=".mp3", sr=16_000, video_start_length=0, load_spec=True):
        self.video_path = Path(video_path)
        self.audio_path = Path(audio_path)
        self.video_start_length = video_start_length

        self.embed_path = None
        self.embed_saved = False
        self.embed = None

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
        from local.loader.constants import EMBED_DIR, SPEC_DIR

        video_name_stem = self.video_path.stem

        embed_dir = Path(EMBED_DIR)
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

if __name__ == "__main__":
    signal = Signal("../../data/train/AvWWVOgaMlk_cropped.mp4", "../../data/train/audio/AvWWVOgaMlk_cropped.mp3")
