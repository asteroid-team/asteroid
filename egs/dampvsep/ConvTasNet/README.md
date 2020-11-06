## Description

ConvTasNet model trained using DAMP-VSEP dataset.
The dataset is preprocessed to obtain only single ensembles performances.
The preprocess return two train sets, one validation and one test sets.

The preprocessing steps can be found is [this repo](https://github.com/groadabike/DAMP-VSEP-Singles)

The details of the dataset:

| Dataset       |   Perf    | hrs       |
|:--------------|----------:|----------:|
| train_english |  9243     |    77     |
| train_singles |  20660    |   174     |
| valid         |  100      |   0.8     |
| test          |  100      |   0.8     |



## Results
The next results were obtained by remixing the sources.
Results using the original mixture are pending.

|               | Mixture   |SI-SNRi(dB) (v)| STOI (v)|SDRi(dB) (b)|
|:-------------:|:---------:|:-------------:|:-------:|:----------:|
| train_english | remix     |        14.3   | 0.6872  |       14.5 |
| train_english | original  |        ---    |   ---   |       ---  |
| train_singles | remix     |        15.0   | 0.6808  |       14.8 |
| train_singles | original  |        ---    |   ---   |       ---  |

(v): vocal
(b): background accompaniment

## Python requirements

pip install librosa
conda install -c conda-forge ffmpeg
