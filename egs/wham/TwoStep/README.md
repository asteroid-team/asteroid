### Description
A two-step training procedure for source
separation via a deep neural network. In the first step we learn a
transform (and it’s inverse) to a latent space where masking-based
separation performance using oracles is optimal. For the second step,
we train a separation module that operates on the previously learned
space.

### Results

|      |   Task    | n_blocks | n_repeats | batch size |SI-SNRi(dB) |
|:----:|:---------:|:--------:|:---------:|:----------:|:----------:|
| Paper| sep_clean |    8     |     4     |     -      |    16.10   |
| Here | sep_clean |    8     |     4     |     -      |    15.23   |

### References
If you use this model, please cite the original work.
```BibTex
@article{tzinis2019two,
  title={Two-Step Sound Source Separation: Training on Learned Latent Targets},
  author={Tzinis, Efthymios and Venkataramani, Shrikant and Wang, Zhepei and Subakan, Cem and Smaragdis, Paris},
  booktitle={ICASSP 2020-2020 IEEE International Conference on
 Acoustics, Speech and Signal Processing (ICASSP)},
  pages={},
  year={2020},
  organization={IEEE}
}
```

and if you like using `asteroid` you can give us a star! :star:
