## This is the official repository for Multi-Decoder DPRNN, published at ICASSP 2021. 
Summary: Multi-Decoder DPRNN deals with source separation with variable number of speakers. It has 98.5% accuracy in speaker number classification, which is much higher than all previous SOTA methods. It also has similar SNR as models trained separately on different number of speakers, but its runtime is constant and independent of the number of speakers. 

paper arxiv link: https://arxiv.org/abs/2011.12022

## Demo
Project page & example output can be found at: https://junzhejosephzhu.github.io/Multi-Decoder-DPRNN/

## Getting Started
To install the requirements, run ```pip install -r requirements.txt```
To run a pre-trained model on your own .wav mixture files, run ```python eval.py --wav_file {file_name.wav} --use_gpu {1/0} --save_folder {folder_name}```
You can use regular expressions for file names. For example, you can run ```python eval.py --wav_file local/*.wav --use_gpu 0 --save_folder {output}```
The default output directory will be ./output, but you can override that with ```--output_dir``` option

## Train your own model
To train the model, edit the file paths in run.sh and execute ```./run.sh --stage 0``` to generate and train the model
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

# Resources
Pretrained mini model and config can be found at: https://huggingface.co/JunzheJosephZhu/MultiDecoderDPRNN \

#### This is the refactored version of the code, with some hyperparameter changes. If you want to reproduce the paper results, original experiment code & config can be found at https://github.com/JunzheJosephZhu/MultiDecoder-DPRNN

Original Paper Results(Confusion Matrix)
2    | 3    | 4    |5
-----|------|------|--
2998 | 17   | 1    |0
2    | 2977 | 27   |0
0    | 6    | 2928 |80
0    | 0    | 44   |2920
