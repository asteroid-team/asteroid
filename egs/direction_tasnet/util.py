import numpy as np

from scipy import signal

def si_sdr(estimated_signal, reference_signals, scaling=True):
    """
    This is a scale invariant SDR. See https://arxiv.org/pdf/1811.02508.pdf
    or https://github.com/sigsep/bsseval/issues/3 for the motivation and
    explanation
    Input:
        estimated_signal and reference signals are (N,) numpy arrays
    Returns: SI-SDR as scalar
    """
    # first align them
    '''
    lag_values = np.arange(-len(estimated_signal)+1, len(estimated_signal))
    crosscorr = signal.correlate(estimated_signal, reference_signals)
    max_crosscorr_idx = np.argmax(crosscorr)
    lag = -lag_values[max_crosscorr_idx]
    if lag>0:
        estimated_signal=estimated_signal[:-lag]
        reference_signals=reference_signals[lag:]
    else:
        estimated_signal=estimated_signal[-lag:]
        reference_signals=reference_signals[:lag]

    plt.figure()
    plt.plot(estimated_signal)
    plt.plot(reference_signals)
    plt.show()
    '''
    
    Rss = np.dot(reference_signals, reference_signals)
    this_s = reference_signals
    
    
    if scaling:
        # get the scaling factor for clean sources
        a = np.dot(this_s, estimated_signal) / Rss
    else:
        a = 1

    e_true = a * this_s
    e_res = estimated_signal - e_true

    Sss = (e_true**2).sum()
    Snn = (e_res**2).sum()

    SDR = 10 * np.log10(Sss/Snn)

    return SDR

def real_bpf(sr, fmin, fmax, fw):
    fw_half=fw//2
    wind=np.zeros((fw,), dtype='complex')
    for i in range(fw):
        j=i-fw_half
        if j!=0:
            wind[i]=np.sin(2*np.pi*j*fmax/sr)/j/np.pi-np.sin(2*np.pi*j*fmin/sr)/j/np.pi
    wind[fw_half]=2*(fmax-fmin)/sr
    s=np.sum(wind)/fw
    wind=wind-s
    return wind

def calculate_gain(original, mixed, beamformed, fs=48000, freq_low=100, freq_high=10000, wind_size=129):
    # first high pass filter
    wind=real_bpf(48000, freq_low, freq_high, wind_size)
    original=np.real(np.convolve(original, wind, 'valid'))
    mixed=np.real(np.convolve(mixed, wind, 'valid'))
    beamformed=np.real(np.convolve(beamformed, wind, 'valid'))

    #ipd.display(ipd.Audio(original, rate=48000))
    #ipd.display(ipd.Audio(mixed, rate=48000))
    #ipd.display(ipd.Audio(beamformed, rate=48000))
    
    l=np.min([len(original), len(mixed), len(beamformed)])
    si_sdr1=si_sdr(mixed[:l], original[:l])
    si_sdr2=si_sdr(beamformed[:l], original[:l])
    return (si_sdr2-si_sdr1, si_sdr1, si_sdr2)

def db(x):
    return 10*np.log10(x)

def dbto(x):
    return 10**(x/10)

def power(signal):
    return np.sum(np.abs(signal)**2)/signal.size

def normalize(signal): # keep a std of 0.1 and min/max of +-1
    return np.clip(signal/np.std(signal)*0.1, -1, 1)

def mix(signal, noise, target_snr_db):
    psig=power(signal)
    pnoi=power(noise)
    newnoise=noise*np.sqrt(dbto(db(psig/pnoi)-target_snr_db))
    res=signal+newnoise
    return res, newnoise


if __name__ == "__main__":
    pass