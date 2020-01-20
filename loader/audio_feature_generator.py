'''
    This file deals with raw audio and converts them into
    spectrogram.
'''
import librosa
import numpy as np


def convert_to_spectrogram(audio_raw: np.ndarray, sr=16_000, hann_length=400, hop_length=160, n_fft=512, p=0.3) -> np.ndarray:
    """
        convert audio to spectrogram

        Args:
            audio_raw: signal in time domain
            sr: sampling rate
            hann_length: hanning length
            hop_length: hop length
            n_fft: n_fft
            p: power value
        Returns:
            spec: spectrogram of audio_raw with specified parameters
    """
    #Calculate the short time fourier transform
    #Appending zeros to get the exact shape
    stereo = False
    if len(audio_raw.shape) > 1 and audio_raw.shape[1] == 2:
        stereo = True
    if stereo:
        audio_raw = audio_raw[:, 0].expand_dims(1)
        audio_raw_right = audio_raw[:, 1].expand_dims(1)

    #audio_raw = np.squeeze(np.concatenate((audio_raw, np.zeros((50, 1)))))
    audio_raw = librosa.util.fix_length(audio_raw, 48050)
    spec = librosa.stft(audio_raw, n_fft=n_fft, hop_length=hop_length, win_length=hann_length, center=False)
    
    if stereo:
        spec_right = librosa.stft(np.squeeze(audio_raw), n_fft=n_fft, hop_length=hop_length, win_length=hann_length, center=False)

    #Power Law Compression, NOTE: Apply power law on complex numbers only
    spec = np.power(spec, p)

    #Unstack Real and Imaginary Components into different dimensions
    #(257, 298) imaginary numbers -> (257, 298, 2) real numbers
    spec = np.dstack((spec.real, spec.imag))

    return spec

def convert_to_wave(spec: np.ndarray, sr=16_000, hann_length=400, hop_length=160, n_fft=512, p=0.3) -> np.ndarray:
    """
        convert spectrogram to wave

        Args:
            spec: spectrogram
            sr: sampling rate
            hann_length: hanning length
            hop_length: hop length
            n_fft: n_fft
            p: power value
        Returns:
            audio_raw: signal in time domain
    """
    
    assert spec.shape[-1] == 2, "Input has to be complex split across dimensions"
    spec = spec[:, :, 0] + 1j*spec[:, :, 1]

    #Inverse Power Law, NOTE: Apply power law on complex numbers only
    spec = np.power(spec, 1/p)

    #Inverse STFT
    audio_raw = librosa.istft(spec, hop_length=hop_length, win_length=hann_length, center=False)
    return audio_raw



if __name__ == "__main__":
    spec = convert_to_spectrogram(librosa.load("../../data/train/audio/0CabGpMJkiY_3453_3240_final.wav", mono=True, sr=16_000, duration=3)[0])
    print(spec)
    print(spec.shape)
    #orig = convert_to_wave(spec)
    #librosa.output.write_wav("hmmm.wav", orig, sr=16_000)

