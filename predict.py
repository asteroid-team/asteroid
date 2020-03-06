import sys

import torch
import librosa
import numpy as np
import pandas as pd
from tqdm import trange

from typing import List
from pathlib import Path
from argparse import ArgumentParser

from src.postprocess import filter_audio
from src.models import Audio_Visual_Fusion as AVFusion
from src.loader import convert_to_wave, convert_to_spectrogram

def _preprocess_audio(audio: np.ndarray):
    if len(audio.shape) == 1:
        audio = convert_to_spectrogram(audio)

    if audio.shape[0] != 2:
        audio = np.expand_dims(audio.transpose(2, 1, 0), axis=0)

    audio = torch.from_numpy(audio)
    return audio

def _preprocess_video(videos: np.ndarray):
    # add code to convert face to embed
    videos = [np.expand_dims(i, axis=0) for i in videos]
    videos = [torch.from_numpy(i) for i in videos]
    return videos

def _to_numpy(tensor: torch.Tensor):
    tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.numpy()

def predict(model: AVFusion, audio: torch.Tensor, videos: torch.Tensor, device: torch.device):
    """
        Predict the separated audio using the trained model

        Args:
            model: trained model
            audio: input mixed audio
            videos: input video embedding of the face
            device: torch device
        Returns:
            spectrograms: predicted separated spectrograms
    """
    audio = audio.to(device)
    videos = [i.to(device) for i in videos]

    output = model(audio, videos)
    output = output.permute(0, 3, 2, 1, 4)

    output = _to_numpy(output)

    spectrograms = []
    for i in range(output.shape[-1]):
        spectrogram = np.squeeze(output[..., i])
        spectrograms.append(spectrogram)
    return spectrograms


def generate_audio(model: AVFusion, audio_path: Path, video_paths: List[Path],
                   device: torch.device, save:bool=True, output_dir:Path=Path("output/"),
                   return_spectrograms:bool=False, postprocess:bool=True):
    """
        Separate and generate audio

        Args:
            model: trained model
            audio_path: path of mixed audio (wav or spectrogram)
            video_paths: path of video embedding
            device: torch device
            save: to save separated audio in `output_dir/`
            output_dir: location to save audio with the same name as video
            return_spectrograms: returns spectrograms, doesn't save the audio
        Returns:
            output_audios: list of numpy array of separated wav
    """
    if audio_path.suffix.endswith("wav"):
        audio = librosa.load(audio_path, sr=16_000)[0]
    elif audio_path.suffix.endswith("npy"):
        audio = np.load(audio_path)
    else:
        raise ValueError("wav or npy only for audio, for now..")

    assert all(i.suffix.endswith("npy") for i in video_paths), "Video embedding only for now"
    videos = [np.load(i) for i in video_paths]

    audio = _preprocess_audio(audio)
    videos = _preprocess_video(videos)

    spectrograms = predict(model, audio, videos, device)
    if return_spectrograms:
        print("Not saving the audio")
        return spectrograms

    output_audios = [convert_to_wave(i) for i in spectrograms]
    if postprocess:
        output_audios = [filter_audio(i) for i in output_audios]

    if not save:
        return output_audios

    for i, path in enumerate(video_paths):
        name = path.stem + ".wav"
        librosa.output.write_wav(output_dir / name, output_audios[i], sr=16_000)
    return output_audios

def _predict_row(model, df, row_idx, device):
    row = df.iloc[row_idx]

    audio = row[-1]
    video = [row[2], row[3]]
    video = [i.replace("audio", "embed").replace("wav", "npy") for i in video]

    audio = Path(audio)
    video = [Path(i) for i in video]

    generate_audio(model, audio, video, device=device, save=True)

if __name__ == "__main__":
    device = torch.device("cuda")
    model =  AVFusion().to(device)
    model.load_state_dict(torch.load("last_full.pth")["model_state_dict"])

    train_df = pd.read_csv("filtered_train.csv")

    for i in trange(1000):
        _predict_row(model, train_df, i, device)
    #generate_audio(model, "../data/train/spec/0.npy", ["../data/train/embed/IlnHVjvBDU0_4750_3041_final_part1.npy", "../data/train/embed/0m4_JnhSoDc_5171_2740_final_part2.npy"], device=device)
    #predict(model, "../data/train/spec/123724.npy", ["../data/train/embed/3fY4X9NlONY_4000_3240_final_part1.npy", "../data/train/embed/M6pEJ3_z5-o_5648_3013_final_part0.npy"])

