### About MetricGAN

MetricGAN's approach aims to optimize the generator with respect to one or multiple
evaluation metrics. The original paper describes in details the architectures used and the 
training procedure is available [here](https://arxiv.org/pdf/1905.04874.pdf)

The original implementation of MetricGAN with tensorflow is available [here](https://github.com/JasonSWFu/MetricGAN).


 
### Results

All the models were trained on [LibriMix](../librimix) train-360 16K with task enh_single.

|   | SI-SNRi(dB) | SI-SNR(dB) | STOI | STOIi | PESQ | PESQi |
|:---------------:|:-----------:|:----------:|:----:|:-----:|:----:|:-----:|
| MetricGAN      | 4.5      | 7.9        |0.86 |0.07  | 1.39 |  0.23 |
| SEGAN           | 4.2       | 7.7      |0.83 |0.03   | 1.37  |  0.21 |