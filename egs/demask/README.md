### DeMask

This recipe shows how can build a simple model for surgical mask speech enhancement and was developed for [Pytorch Summer Hackathon 2020](https://devpost.com/software/DeMask).

DeMask is a simple, yet effective, end-to-end model to enhance speech when wearing face masks.
It restores the frequency content which is distorted by the face mask,
making the muffled speech sound cleaner.

#### DeMask's inspiration

We were having a fast call and one of us was in the train.
We couldn't hear him well because he was wearing a mask.
We directly thought about building surgically masked speech enhancement model with
Asteroid!
In the current covid pandemic situation, we can build better mask-adapted
speech technologies to help people keep their masks on and spread their words without
spreading the virus.

#### Recipe description

##### Synthetic Dataset creation
The most challenging part for doing surgical mask speech enhancement was finding a suitable dataset to train
a DNN enhancement model on.
After some attempts we resorted to use the frequency responses measured in [1] (measured from real world mask) to
create a synthetic dataset for mask speech enhancement.

Such dataset class is in `local/demask_dataset.py`.

We use LibriSpeech [2] for clean speech.
 To achieve better generalization to real world scenarios we
also apply, directly to clean speech, data augmentation. We use speed perturbation
and we randomly change the clean speech utterance amplitude.
We also apply reverberation with synthetic Room Impulse Responses from FUSS dataset [3].

```python
def augment_clean(self, clean):

    speed = eval(self.configs["training"]["speed_augm"])
    c_gain = eval(self.configs["training"]["gain_augm"])

    fx = AudioEffectsChain().speed(speed)  # speed perturb
    clean = fx(clean)

    if self.rirs:
        c_rir = np.random.choice(self.rirs, 1)[0]
        c_rir, fs = sf.read(c_rir)
        assert fs == self.configs["data"]["fs"]
        clean = fftconvolve(clean, c_rir)

    fx = AudioEffectsChain().custom("norm {}".format(c_gain))  # random gain
    clean = fx(clean)

    return clean, c_gain
```

we then apply a mask-simulating FIR filter to such augmented clean speech.

The FIR filter frequency response was taken directly from [1] but, to account for real world variations, we randomly perturb
the frequency response with gaussian noise and use `scipy.signal.firwin2` design the filter:

```python
if self.train:
    # augment the gains with random noise: no mask is created equal
    snr = 10 ** (eval(self.configs["training"]["gaussian_mask_noise_snr_dB"]) / 20)
    gains += np.random.normal(0, np.var(gains) / snr, gains.shape)

    fir = firwin2(
        self.configs["training"]["n_taps"], freqs, gains, fs=self.configs["data"]["fs"]
    )
```

such filter Impulse Response (IR) is then convolved with the augmented clean speech utterance to obtain a simulated masked signal.
We then carefully realign the phase of the original augmented clean speech signal and the masked one:

```python
masked = fftconvolve(clean, fir)
clean = np.pad(clean, ((len(fir) - 1) // 2, 0), mode="constant")
trim_start = (len(fir) - 1) // 2
trim_end = len(clean) - len(fir) + 1
clean = clean[trim_start:trim_end]
masked = masked[trim_start:trim_end]
```

Finally, after padding we add optional real-world noise or more simply, gaussian noise to both masked and clean target speech,
in order to make the model more robust to enviromental noise.

In an actual application one would do both Speech Enhancement and De-Masking at the same time.
We choose to do only De-Masking here to show that such thing is possible and easy to set up in Asteroid.
In that case the target would be the clean speech without the additive enviromental noise.

##### Model

The DeMask model is very simple and can be found in `asteroid.models.demask`.
We choose a simple magnitude Short-Time-Fourier-Transform (STFT) masking approach because we found STFT to generalize better to
real-world examples. But in the model we give the user the opportunity to explore other configurations, such for example
both complex and real part STFT masking:

```python
if self.output_type == "mag":
    masked_tf_rep = est_masks.repeat(1, 2, 1) * tf_rep
elif self.output_type == "reim":
    masked_tf_rep = est_masks * tf_rep.unsqueeze(1)
else:
    raise NotImplementedError
```

the Neural Network itself is a simple MLP with just one hidden-layer as we found this sufficient to give satisfactory results.
The MLP works can work directly in a frame-wise fashion because the simulated mask frequency response is stationary.
This means this model can work in real-time.


```python
net = [norms.get(norm_type)(n_feats_input)]
in_chan = n_feats_input
for layer in range(len(hidden_dims)):
    net.extend(
        [
        nn.Conv1d(in_chan, hidden_dims[layer], 1),
        norms.get(norm_type)(hidden_dims[layer]),
        activations.get(activation)(),
        nn.Dropout(dropout),
        ]
        )
        in_chan = hidden_dims[layer]

net.extend([nn.Conv1d(in_chan, n_feats_output, 1), activations.get(mask_act)()])

self.masker = nn.Sequential(*net)
```



**References**

[1] Corey, Ryan M., Uriah Jones, and Andrew C. Singer. "Acoustic effects of medical, cloth, and transparent face masks on speech signals." arXiv preprint arXiv:2008.04521 (2020).

[2] Panayotov, Vassil, et al. "Librispeech: an asr corpus based on public domain audio books." 2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2015.

[3] Scott Wisdom, Hakan Erdogan, Daniel P. W. Ellis, Romain Serizel, Nicolas Turpault, Eduardo Fonseca, Justin Salamon, Prem Seetharaman, and John R. Hershey. What's all the fuss about free universal sound separation data? In in preparation. 2020.
