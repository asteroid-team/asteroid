### The CHiME-4 dataset

The CHiME-4 dataset is part of the 4th CHiME speech separation and recognition challenge.

It was released in 2016 and revisits the datasets originally recorded for CHiME-3.

All data and information are available [here](http://spandh.dcs.shef.ac.uk/chime_challenge/CHiME4/index.html).

For now, this recipe only deals with the `real_1_ch_track` part of the dataset. 
As the channel to use for the training set wasn't defined by 
the challenge's rules, we will set it randomly.

NOTE : 
This dataset uses real noisy data. This means the clean speech from the noisy
utterances is not available. This makes it not suitable for the usual training 
procedure.



**References**
~~~BibTeX
@article{vincent:hal-01399180,
  TITLE = {{An analysis of environment, microphone and data simulation mismatches in robust speech recognition}},
  AUTHOR = {Vincent, Emmanuel and Watanabe, Shinji and Nugraha, Aditya Arie and Barker, Jon and Marxer, Ricard},
  URL = {https://hal.inria.fr/hal-01399180},
  JOURNAL = {{Computer Speech and Language}},
  PUBLISHER = {{Elsevier}},
  VOLUME = {46},
  PAGES = {535-557},
  YEAR = {2017},
  MONTH = Jul,
  DOI = {10.1016/j.csl.2016.11.005},
  KEYWORDS = {speech enhancement ; Robust ASR ; train/test mismatch ; microphone array},
  PDF = {https://hal.inria.fr/hal-01399180/file/vincent_CSL16.pdf},
  HAL_ID = {hal-01399180},
  HAL_VERSION = {v1},
}
~~~