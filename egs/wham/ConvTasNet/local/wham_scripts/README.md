# Generating the WHAM! dataset

## Python requirements

These scripts require Python 3, and the Numpy, Scipy, Pandas, and Pysoundfile packages.

## Prerequisites

This requires the wsj0 (https://catalog.ldc.upenn.edu/LDC93S6A/) dataset,
and the WHAM noise corpus.

Alternatively, if you have already built the wsj0-2mix dataset, i.e. the output from the Matlab script *create_wav_2speakers.m*
available from 
*http://www.merl.com/demos/deep-clustering/create-speaker-mixtures.zip*
you can create WHAM! from that.

## Creating WHAM! from scratch

```sh
$ python create_wham_from_scratch.py 
    --wsj0-root  /path/to/the/wsj/dataset/
    --wham-noise-root /path/to/wham_noise/ 
    --output-dir /path/to/output/directory/
 
```

The arguments for the script are:
* **wsj0-root**:  Path to the folder containing `wsj0/`
* **wham-noise-root**: Folder where the unzipped `wham_noise` was downloaded.
* **output-dir**: Where to write the new dataset.  This will write about 243 GB of data.

## Creating WHAM! from wsj0-2mix

```sh
$ python create_wham_from_wsjmix.py 
    --wsjmix-dir-16k /path/to/the/16k/wsj/dataset/
    --wsjmix-dir-8k  /path/to/the/8k/wsj/dataset/
    --wham-noise-root /path/to/wham_noise/
    --output-dir /path/to/output/directory/
 
```
The arguments for the script are:

* **wsjmix-dir-16k**: Folder containing original wsj0-2mix 2speakers 16 kHz dataset. Input argument
  from `create_wav_2speakers.m` Matlab script
* **wsjmix-dir-8k**: Folder containing original wsj0-2mix 2speakers 8 kHz dataset. Input argument
  from `create_wav_2speakers.m` Matlab script
* **wham-noise-root**: Folder where the unzipped `wham_noise` was downloaded.
* **output-dir**: Where to write the new dataset.  This will write about 243 GB of data.

## Output Data

The script outputs all possible permutations of the dataset (8 kHz and 16 kHz sampling rate plus max and min style utterance truncation)

For each of the training (tr), validation (cv), and testing (tt) sets mixtures are created for three possible experiments:
1. `mix_single`: for speech enhancement, single speaker in noise

2. `mix_clean`: clean speech separation for two speakers.  The relative levels between speakers
should match the original wsj0-2mix dataset, but the overall level of the mix will be different.

3. `mix_both`: contains mixtures of both speakers and noise

The isolated source are in the s1, s2, and noise directories 

## Creating wsj0-2mix

We also include a python script that replicates the output of the Matlab script *create_wav_2speakers.m*
available from 
*http://www.merl.com/demos/deep-clustering/create-speaker-mixtures.zip* for those interested in only the clean speech separation dataset. 

```sh
$ python create_wav_2speakers.py 
    --wsj0-root  /path/to/the/wsj/dataset/
    --wham-noise-root /path/to/wham_noise/
    --output-dir /path/to/the/output/directory/
 
```
The arguments for the script are:
* **wsj0-root**:  Path to the folder containing `wsj0/`
* **wham-noise-root**: Folder where the unzipped `wham_noise` was downloaded.
* **output-dir**: Where to write the new dataset.  This will write about 243 GB of data.
