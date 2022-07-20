## This is the repository for Multi-Decoder DPRNN, published at ICASSP 2021. 
Summary: Multi-Decoder DPRNN deals with source separation with variable number of speakers. It has 98.5% accuracy in speaker number classification, which is much higher than all previous SOTA methods. It also has similar SNR as models trained separately on different number of speakers, but its runtime is constant and independent of the number of speakers. 

paper link: https://arxiv.org/abs/2011.12022

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



Pretrained mini model and config can be found at: https://huggingface.co/JunzheJosephZhu/MultiDecoderDPRNN \
Project page & example output can be found at: https://junzhejosephzhu.github.io/Multi-Decoder-DPRNN/
#### This is the refactored version of the code for ease of production use. If you want to reproduce the paper results, original experiment code & config can be found at https://github.com/JunzheJosephZhu/MultiDecoder-DPRNN

Original Paper Results(Confusion Matrix)
2    | 3    | 4    |5
-----|------|------|--
2998 | 17   | 1    |0
2    | 2977 | 27   |0
0    | 6    | 2928 |80
0    | 0    | 44   |2920
