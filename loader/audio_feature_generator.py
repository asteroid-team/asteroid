'''
    This file deals with raw audio and converts them into
    spectrogram.
'''
import librosa
import numpy as np


def convert_to_spectrogram(audio_raw, sr=16_000, hann_length=400, hop_length=160, n_fft=512, p=0.3):
    #Calculate the short time fourier transform
    spec = librosa.stft(audio_raw, n_fft=n_fft, hop_length=hop_length, win_length=hann_length, center=False) 

    #Power Law Compression, NOTE: Apply power law on complex numbers only
    spec = np.power(spec, p)

    #Unstack Real and Imaginary Components into different dimensions
    #(257, 297) imaginary numbers -> (257, 297, 2) real numbers
    spec = np.dstack((spec.real, spec.imag))

    return spec

def convert_to_wave(spec, sr=16_000, hann_length=400, hop_length=160, n_fft=512, p=0.3):
    assert spec.shape[-1] == 2, "Input has to be complex split across dimensions"
    spec = spec[:, :, 0] + 1j*spec[:, :, 1]

    #Inverse Power Law, NOTE: Apply power law on complex numbers only
    spec = np.power(spec, 1/p)

    #Inverse STFT
    audio_raw = librosa.istft(spec, hop_length=hop_length, win_length=hann_length, center=False)
    return audio_raw



if __name__ == "__main__":
    spec = convert_to_spectrogram(librosa.load("../../data/train/audio/AvWWVOgaMlk_cropped.wav", sr=16_000, duration=3)[0])
    orig = convert_to_wave(spec)
    librosa.output.write_wav("hmmm.wav", orig, sr=16_000)
