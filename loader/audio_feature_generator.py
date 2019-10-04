'''
    This file deals with raw audio and converts them into
    spectrogram.
'''
import librosa
import numpy as np

def convert_to_spectrogram(audio_raw, sr=16_000, hann_length=400, hop_length=160, n_fft=512, p=0.3):
    #Calculate the short time fourier transform
    spec = librosa.stft(audio_raw, n_fft=n_fft, hop_length=hop_length, win_length=hann_length, center=False) 

    #Unstack Real and Imaginary Components into different dimensions
    #(257, 297) imaginary numbers -> (257, 297, 2) real numbers
    spec = np.dstack((spec.real, spec.imag))

    #Power Law Compression
    spec = spec ** (p)

    return spec


if __name__ == "__main__":
    print(convert_to_spectrogram(librosa.load("../../data/train/mixed/0.wav", sr=16_000, duration=3)[0]).shape)
