### Results

|      |   task    |kernel size|chunk size|batch size|SI-SNRi(dB) | SDRi(dB)|
|:----:|:---------:|:---------:|:--------:|:--------:|:----------:|:-------:|
| Paper| sep_clean |    16     |     100  |     -    |    15.9    |   16.1  |
| Here | sep_clean |    16     |     100  |     8    |    17.7    |  18.0   |
| Paper| sep_clean |    2     |    250   |     -    |    18.8    |  19.0   |
| Here | sep_clean |    2     |     250  |     3   |    19.3    |  19.5   |

Both models with ks=16 and ks=2 were trained with segments of 2seconds only.