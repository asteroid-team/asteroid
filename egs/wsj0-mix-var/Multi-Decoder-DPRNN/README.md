## This is the official repository for Multi-Decoder DPRNN, published at ICASSP 2021. 
**Summary**: Multi-Decoder DPRNN deals with source separation with variable number of speakers. It has 98.5% accuracy in speaker number classification, which is much higher than all previous SOTA methods. It also has similar SNR as models trained separately on different number of speakers, but **its runtime is constant and independent of the number of speakers.**

**Abstract**: We propose an end-to-end trainable approach to single-channel speech separation with unknown number of speakers, **only training a single model for arbitrary number of speakers**. Our approach extends the MulCat source separation backbone with additional output heads: a count-head to infer the number of speakers, and decoder-heads for reconstructing the original signals. Beyond the model, we also propose a metric on how to evaluate source separation with variable number of speakers. Specifically, we cleared up the issue on how to evaluate the quality when the ground-truth hasmore or less speakers than the ones predicted by the model. We evaluate our approach on the WSJ0-mix datasets, with mixtures up to five speakers. **We demonstrate that our approach outperforms state-of-the-art in counting the number of speakers and remains competitive in quality of reconstructed signals.**

paper arxiv link: https://arxiv.org/abs/2011.12022

## Project Page & Examples
Project page & example output can be found [here](https://junzhejosephzhu.github.io/Multi-Decoder-DPRNN/)

## Getting Started
### Colab notebooks:
* Usage Example: [![Usage Example](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11MGx3_sgOrQrB6k8edyAvg5mGIxqR5ED?usp=sharing)
### Run locally
To Setup, Run the following commands:
```
git clone https://github.com/asteroid-team/asteroid.git
cd asteroid/egs/wsj0-mix-var/Multi-Decoder-DPRNN
pip install -r requirements.txt
```
To test run on a wav file, run:
```
python separate.py --wav_file ${mixture_file}
```
To load the model, run:
```
from model import MultiDecoderDPRNN
model = MultiDecoderDPRNN.from_pretrained("JunzheJosephZhu/MultiDecoderDPRNN").eval()
model.separate(input_tensor)
```

Install asteroid by running ```pip install -e .``` in asteroid directory
To install the requirements, run ```pip install -r requirements.txt```

To run a pre-trained model on your own .wav mixture files, run ```python eval.py --wav_file {file_name.wav} --use_gpu {1/0}```. The script should automatically download a pre-trained model(link below).

You can use regular expressions for file names. For example, you can run ```python eval.py --wav_file local/*.wav --use_gpu 0 ```

The default output directory will be ./output, but you can override that with ```--output_dir``` option

If you want to download an alternative pre-trained model, you can create a folder, and save the pretrained model in ```{folder_name}/checkpoints/best-model.ckpt```, then run ```python eval.py --wav_file {file_name.wav} --use_gpu {1/0} --exp_dir {folder_name}```

## Train your own model
To train the model, edit the file paths in run.sh and execute ```./run.sh --stage 0```, follow the instructions to generate dataset and train the model.

After training the model, execute ```./run.sh --stage 4``` to evaluate the model. Some examples will be saved in exp/tmp_uuid/examples

## Kindly cite this paper
```
@INPROCEEDINGS{9414205,
  author={Zhu, Junzhe and Yeh, Raymond A. and Hasegawa-Johnson, Mark},
  booktitle={ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Multi-Decoder Dprnn: Source Separation for Variable Number of Speakers}, 
  year={2021},
  volume={},
  number={},
  pages={3420-3424},
  doi={10.1109/ICASSP39728.2021.9414205}}
```

## Resources
Pretrained mini model and config can be found at: https://huggingface.co/JunzheJosephZhu/MultiDecoderDPRNN \

This is the refactored version of the code, with some hyperparameter changes. If you want to reproduce the paper results, original experiment code & config can be found at https://github.com/JunzheJosephZhu/MultiDecoder-DPRNN

**Original Paper Results**(Confusion Matrix)
2    | 3    | 4    |5
-----|------|------|--
2998 | 17   | 1    |0
2    | 2977 | 27   |0
0    | 6    | 2928 |80
0    | 0    | 44   |2920

## Contact the author
If you have any question, you can reach me at josefzhu@stanford.edu
