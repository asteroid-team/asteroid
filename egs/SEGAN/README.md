### About SEGAN

SEGAN stands for Speech Enhancement Generative Adversarial Network.
This approach is a 1D adaptation of [pix2pix](https://arxiv.org/pdf/1611.07004.pdf).
A detailed description of the generator, the discriminator and the training process 
is available [here](https://arxiv.org/pdf/1703.09452.pdf).

The original implementation of SEGAN with tensorflow is available [here](https://github.com/santi-pdp/segan).
The implementation used to reproduce SEGAN in this recipe was adapted from [here](https://github.com/dansuh17/segan-pytorch).

### Investigation

In this recipe, we investigate several aspects of SEGAN.
The community has already reported that the noise that is added the compressed 
representation of the signal is ignored by the generator in the reconstruction phase
[[1]](https://arxiv.org/pdf/1711.05747.pdf) [[2]](https://arxiv.org/pdf/1511.05440.pdf).
We have been able to confirm and reproduce this result
Furthermore, we have investigated the impact of the adversarial loss over the L1
regularization. We found out that removing the adversarial part does not degrade performance 
and using SNR regularization instead of L1 regularization gives better performance

### Results
 
All the models were trained on [LibriMix](../librimix) train-360 with task enh_single.

|  Generator loss | SI-SNRi(dB) | SI-SNR(dB) | STOI | STOIi | PESQ | PESQi |
|:---------------:|:-----------:|:----------:|:----:|:-----:|:----:|:-----:|
| ConvTasnet      | 10.5        | 13.9        |0.92 |0.12   | 2.1  |  0.91 |
| Adversarial + L1| 4.2         | 7.66        |0.83 |0.03   | 1.37  |  0.21 |
|  L1             | 4.1         | 7.59        |0.83  |0.03  |1.39  |  0.23 |
| SNR             | 4.6         | 8.0         |0.84  |0.05  |1.5   |  0.3  |
