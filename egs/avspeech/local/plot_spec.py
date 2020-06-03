#TODO: make it compatible with current changes
import sys

import librosa
import librosa.display

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from predict import generate_audio
from src.models import Audio_Visual_Fusion as AVFusion
from src.loader import convert_to_spectrogram

def _plot(i, spec, title):
    if len(spec.shape) == 3:
        spec = spec[:, :, 0] + 1j*spec[:, :, 1]
    plt.subplot(3, 2, i)
    librosa.display.specshow(spec)
    plt.title(title)

def plot_all(mixed_spec, pred_first_spec, pred_second_spec, true_first_spec, true_second_spec):
    _plot(1, pred_first_spec, "First Prediction")
    _plot(2, true_first_spec, "True First")
    _plot(3, pred_second_spec, "Second Prediction")
    _plot(4, true_second_spec, "True Second")
    _plot(5, mixed_spec, "Mixed Input")

    plt.show()

def plot(spec):
    plt.subplot(3, 1, 1)
    spec = spec[:, :, 0] + 1j*spec[:, :, 1]

    librosa.display.specshow(spec)
    plt.show()

def plot_row(model, df, row_idx, device):
    row = df.iloc[row_idx]

    mixed_spec = convert_to_spectrogram(librosa.load(row[-1], sr=16_000)[0])
    first_spec = convert_to_spectrogram(librosa.load(row[2], sr=16_000)[0])
    second_spec = convert_to_spectrogram(librosa.load(row[3], sr=16_000)[0])

    audio = row[-1]
    video = [row[2], row[3]]
    video = [i.replace("audio", "embed").replace("wav", "npy") for i in video]

    audio = Path(audio)
    video = [Path(i) for i in video]

    output_audios = generate_audio(model, audio, video, device=device, save=False, return_spectrograms=True)

    first = output_audios[0]
    second = output_audios[1]

    plot_all(mixed_spec, first, second, first_spec, second_spec)

if __name__ == "__main__":
    import torch
    from tqdm import trange
    from pathlib import Path
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--file", default=Path("train.csv"), type=Path)
    parser.add_argument("--n", default=3, type=int)
    parser.add_argument("--model-path", default=Path("last_full.pth"), type=Path)

    args = parser.parse_args()

    df = pd.read_csv(args.file)
    device = torch.device("cuda")

    model = AVFusion().to(device)
    model.load_state_dict(torch.load(args.model_path)["model_state_dict"])

    for i in trange(args.n):
        plot_row(model, df, i, device) 
