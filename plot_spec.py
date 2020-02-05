import sys
sys.path.extend(["loader", "models"])

import librosa
import librosa.display

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from predict import generate_audio
from models import Audio_Visual_Fusion as AVFusion
from audio_feature_generator import convert_to_spectrogram

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
    print(row)

    mixed_spec = np.load(row[-1].replace("mixed", "spec").replace("wav", "npy"))
    first_spec = np.load(row[2].replace("audio", "spec").replace("wav", "npy"))
    second_spec = np.load(row[3].replace("audio", "spec").replace("wav", "npy"))

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

    df = pd.read_csv("filtered_val.csv")
    device = torch.device("cuda")

    model = AVFusion().to(device)
    model.load_state_dict(torch.load("last_full.pth")["model_state_dict"])

    for i in range(10):
        plot_row(model, df, i, device) 
    #mixed_first_already_spec = np.load("../../data/train/spec/0.npy")

    #mixed = librosa.load("../../data/train/mixed/0.wav", sr=16000)
    #mixed, sr = mixed[0], mixed[1]
    #print(sr)
    #mixed = convert_to_spectrogram(mixed)

    #true_first = librosa.load("../../data/train/audio/IlnHVjvBDU0_4750_3041_final_part1.wav", sr=16000)[0]
    #true_second = librosa.load("../../data/train/audio/0m4_JnhSoDc_5171_2740_final_part2.wav", sr=16000)[0]
    #
    #true_first_already_spec = np.load("../../data/train/spec/1.npy")

    #true_first = convert_to_spectrogram(true_first)
    #true_second = convert_to_spectrogram(true_second)

    #first_pred = librosa.load("/tmp/first.wav", sr=16000)
    #second_pred = librosa.load("/tmp/second.wav", sr=16000)

    #first_pred = convert_to_spectrogram(first_pred[0])
    #second_pred = convert_to_spectrogram(second_pred[0])

    #first_pred = first_pred[:, :, 0] + 1j*first_pred[:, :, 1]
    #second_pred = second_pred[:, :, 0] + 1j*second_pred[:, :, 1]
    #mixed = mixed[:, :, 0] + 1j*mixed[:, :, 1]

    #mixed_first_already_spec = mixed_first_already_spec[:, :, 0] + 1j*mixed_first_already_spec[:, :, 1]

    #true_first = true_first[:, :, 0] + 1j*true_first[:, :, 1]
    #true_second = true_second[:, :, 0] + 1j*true_second[:, :, 1]

    #plot_all(mixed, first_pred, second_pred, true_first, true_second)

    #spec = np.load("/tmp/first_spec.npy")
    #
    #print(spec.shape)
    #plot(spec)
    #spec = np.load("/tmp/second_spec.npy")
    #
    #print(spec.shape)
    #plot(spec)
