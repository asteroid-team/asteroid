### Description
This simple recipe demonstrates MixIT Unsupervised Separation [1]
on WSJ0-2Mix (we use WHAM clean) with DPRNN.
We use MixIT to train DPRNN on mixtures of mixtures of always two speakers.
Test and validation are the plain WHAM clean with always two speakers.
Results can be improved by not having always 2 speakers in each mixture in
train so that the mixture of mixtures will have not always 4 speakers as shown by [1].



References:

```BibTeX
@article{wisdom2020unsupervised,
  title={Unsupervised sound separation using mixtures of mixtures},
  author={Wisdom, Scott and Tzinis, Efthymios and Erdogan, Hakan and Weiss, Ron J and Wilson, Kevin and Hershey, John R},
  journal={arXiv preprint arXiv:2006.12701},
  year={2020}
}
