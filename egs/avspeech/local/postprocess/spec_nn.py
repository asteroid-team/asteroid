import sys
sys.path.append("../loader/")

import librosa
import librosa.display

import numpy as np
import matplotlib.pyplot as plt

from audio_feature_generator import convert_to_spectrogram, convert_to_wave


def apply_nn(spec):
   plt.subplot(2, 1, 1)
   librosa.display.specshow(spec[..., 0] + 1j*spec[..., 1])
   plt.title("original spec")

   plt.subplot(2, 1, 2)

   rec = librosa.segment.recurrence_matrix(spec, mode='affinity',
                                         metric='cosine', sparse=True)
   spec_nn = librosa.decompose.nn_filter(spec, aggregate=np.average, rec=rec)
   librosa.display.specshow(spec_nn[..., 0] + 1j*spec_nn[..., 1])
   plt.title("nn filtered spec")

   plt.show()

   print(np.sum(spec-spec_nn)**2)


if __name__ == "__main__":
    ##spec = np.load("../../data/train/spec/0.npy")
    #wav = librosa.load("../output/0m4_JnhSoDc_5171_2740_final_part1.wav", sr=16_000)[0]
    #spec = convert_to_spectrogram(wav)
    #spec = librosa.
    #apply_nn(spec)


    ##https://librosa.github.io/librosa_gallery/auto_examples/plot_vocal_separation.html

    y, sr = librosa.load("../output/0m4_JnhSoDc_5171_2740_final_part1.wav", sr=16_000)
    spec = convert_to_spectrogram(y)
    spec = spec[..., 0] + 1j*spec[..., 1]
    S_full, phase = librosa.magphase(spec)

    idx = slice(*librosa.time_to_frames([0, 3], sr=sr))
    print(idx)

    S_filter = librosa.decompose.nn_filter(S_full,
                                           aggregate=np.median,
                                           metric='cosine',
                                           width=int(librosa.time_to_frames(2, sr=sr)))
    S_filter = np.minimum(S_full, S_filter)

    margin_i, margin_v = 2, 10
    power = 2

    mask_i = librosa.util.softmask(S_filter,
                                   margin_i * (S_full - S_filter),
                                   power=power)

    mask_v = librosa.util.softmask(S_full - S_filter,
                                   margin_v * S_filter,
                                   power=power)

    # Once we have the masks, simply multiply them with the input spectrum
    # to separate the components

    S_foreground = mask_v * S_full
    S_background = mask_i * S_full

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(S_full, ref=np.max),
                             y_axis='log', sr=sr)
    plt.title('Full spectrum')
    plt.colorbar()

    plt.subplot(3, 1, 2)
    bg = librosa.amplitude_to_db(S_background, ref=np.max)
    librosa.display.specshow(bg, y_axis='log', sr=sr)
    plt.title('Background')
    plt.colorbar()

    plt.subplot(3, 1, 3)
    fg = librosa.amplitude_to_db(S_foreground, ref=np.max)
    librosa.display.specshow(fg, y_axis='log', x_axis='time', sr=sr)
    plt.title('Foreground')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

    print(bg.shape, fg.shape)
    fg = np.dstack((fg.real, fg.imag))
    bg = np.dstack((bg.real, bg.imag))
    print(bg.shape, fg.shape)
    bg_wav = convert_to_wave(bg)
    fg_wav = convert_to_wave(fg)

    print(bg_wav.shape, fg_wav.shape)

    librosa.output.write_wav("fg.wav", fg_wav, sr=16_000)
    librosa.output.write_wav("bg.wav", bg_wav, sr=16_000)
