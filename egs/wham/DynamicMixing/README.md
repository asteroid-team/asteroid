### WHAM/WSJ0-2MIX Dynamic Mixing

- introduced in [WaveSplit]() [1]
- we added speed perturbation with SoX via [python-audio-effects](https://github.com/carlthome/python-audio-effects).
- this recipe comes with DPRNN as default model but the model can be swapped by modifying model.py and local/conf.yml. 
- Original WSJ0 data is needed to run this recipe. 

### Results:

|   model   |   task    |kernel size|chunk size|batch size|SI-SNRi(dB) | SDRi(dB)|
|:----:|:---------:|:---------:|:--------:|:--------:|:----------:|:-------:|
| DPRNN + DM | sep_clean |    16     |     100  |     8    |    18.4    |  18.64  |
| DPRNN | sep_clean |   16     |     100  |     8    |    17.7    |  17.9   |

--- 
#### References:

[1] 
