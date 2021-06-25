#  CrossNet-Open-Unmix (X-UMX)

This recipe contains __CrossNet-Open-Unmix (X-UMX)__, an improved version of [Open-Unmix (UMX)](https://github.com/sigsep/open-unmix-nnabla) for music source separation. X-UMX achieves an improved performance without additional learnable parameters compared to the original UMX model. Details of X-UMX can be found in [this paper](https://arxiv.org/abs/2010.04228). X-UMX is one of the two official baseline models for the [Music Demixing (MDX) Challenge 2021](https://www.aicrowd.com/challenges/music-demixing-challenge-ismir-2021).

__Related Projects:__ [umx-pytorch](https://github.com/sigsep/open-unmix-pytorch) | [umx-nnabla](https://github.com/sigsep/open-unmix-nnabla) | x-umx-pytorch | [x-umx-nnabla](https://github.com/sony/ai-research-code/tree/master/x-umx) | [musdb](https://github.com/sigsep/sigsep-mus-db) | [museval](https://github.com/sigsep/sigsep-mus-eval)

### Source separation with pretrained model
Pretrained models on MUSDB18 for X-UMX, which reproduce the results from our paper, are available and can be easily tried out:
```
python eval.py --no-cuda --root [Path to MUSDB18]
```
The separations along with the evaluation scores will be saved in `./results_using_pre-trained`.

Please note that X-UMX requires quite some memory due to its crossing architecture. Hence, switching on `--no-cuda` to prevent out-of-memory error is recommended.


### Results on MUSDB18

| Median of Median |   SDR   |   SIR  |  ISR   |  SAR  |
|:----------------:|:-------:|:------:|:------:|:------|
|      vocals      |  6.612  | 14.167 | 11.774 | 6.750 |
|      drums       |  6.471  | 12.677 | 11.276 | 5.968 |
|      bass        |  5.426  | 11.555 | 9.989  | 6.350 |
|      other       |  4.639  | 7.055  | 9.651  | 4.833 |

|  Mean of Median  |   SDR   |   SIR  |  ISR   |  SAR  |
|:----------------:|:-------:|:------:|:------:|:------|
|      vocals      |  3.361  | 8.535  | 10.364 | 5.670 |
|      drums       |  5.793  | 11.167 | 10.164 | 5.687 |
|      bass        |  4.558  | 8.797  | 9.786  | 5.828 |
|      other       |  4.348  | 6.952  | 9.402  | 4.849 |
