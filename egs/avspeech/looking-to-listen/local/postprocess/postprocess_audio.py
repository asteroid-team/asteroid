import scipy.signal as sg
from pysndfx import AudioEffectsChain


def filter_audio(y, sr=16_000, cutoff=15_000, low_cutoff=1, filter_order=5):
    sos = sg.butter(
        filter_order,
        [low_cutoff / sr / 2, cutoff / sr / 2],
        btype="band",
        analog=False,
        output="sos",
    )
    filtered = sg.sosfilt(sos, y)

    return filtered


def shelf(y, sr=16_000, gain=5, frequency=500, slope=0.5, high_frequency=7_000):
    afc = AudioEffectsChain()
    fx = afc.lowshelf(gain=gain, frequency=frequency, slope=slope).highshelf(
        gain=-gain, frequency=high_frequency, slope=slope
    )

    y = fx(y, sample_in=sr, sample_out=sr)

    return y
