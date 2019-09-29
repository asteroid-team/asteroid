import os
import cv2
import librosa
from pathlib import Path


class Signal:
    '''
        This class holds the video frames and the audio signal.

    '''

    def __init__(self, path_to_video, audio_ext=".mp3", sr=44_100):
       self.video_path = Path(path_to_video)

       vid_dir = self.video_path.parents[0]
       video_name = self.video_path.stem

       self.audio_path = Path(os.path.join(vid_dir, "audio", video_name + audio_ext))

       self._load(sr=sr)


    def _load(self, sr):
        self.audio, sr = librosa.load(self.audio_path.as_posix(), sr=sr)
        cap = cv2.VideoCapture(self.video_path.as_posix())


class Augment:
    
    @staticmethod
    def overlay(sig1, sig2, start, end, sr=44_100, w1=0.5, w2=0.5):
        '''
            Overlay sig2 on sig1 at [start, end]
        '''
        #Normalise seconds to frames
        start *= sr
        end *= sr

        len1 = len(sig1)
        len2 = len(sig2)

        sig1[start: start + end] = (w1 * sig1[start: start + end] + w2 * sig2).astype(sig1.dtype)
        return sig1

    @staticmethod
    def combine(*signals, weights=None):
        if len(signals) == 0:
            return None

        total_signals = len(signals)
        
        if weights is None:
            weights = np.ones((total_signals, 1), dtype=np.float32) / total_signals

        combined_signal = np.zeros((signals[0].shape[0], 1), dtype=np.float32)
        weight = 0
        
        for i, w in enumerate(weights):
            combined_signal += signals[i] * w
            weight += w
        
        return combined_signal

    @staticmethod
    def align(main_signal, all_signals, all_alignments, sr=44_100):
        '''
            Align signals of different length (smaller) to main_signal.
            Alignments are tuples containing start and end points wrt main_signal.
        '''

        length = len(main_signal)
        assert len(all_signals) == len(all_alignments), "All signals should have alignments"

        for i, (signal, alignment) in enumerate(zip(all_signals, all_alignments)):
            start, end = alignment
            start, end = start * sr, end * sr
            
            prefix = np.zeros((start, 1))
            suffix = np.zeros((length - end, 1))

            signal = np.concatenate((prefix, signal, suffix), axis=0)

            all_signals[i] = signal
            


if __name__ == "__main__":
    signal = Signal("../../data/train/AvWWVOgaMlk_cropped.mp4")
