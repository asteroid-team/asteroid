# Asteroid : Audio Source Separation on steroids
[![Build Status][travis-badge]][travis]
[![codecov][codecov-badge]][codecov]

[![Slack][slack-badge]][slack-invite]

Asteroid is a Pytorch-based source separation and speech enhancement 
API that enables fast experimentation on common datasets. 
It comes with a source code written to support a large range 
of datasets, architectures, loss functions etc... and a set of
 recipes to reproduce some important papers.  
Asteroid is intended to be a __community-based project__ 
so hop on and help us !

### You use asteroid or you want to? 
Please, if you have found a bug, [open an issue][issue], 
if you solved it, [open a pull request][pr] !  
Same goes for new features, tell us what you want or help us building it !  
Don't hesitate to [join the slack][slack-invite] 
and ask questions / suggest new features there as well !
## Table of contents
- [Installation](#installation)
- [Tutorials](#highlights)
- [Running a recipe](#running-a-recipe)
- [Available recipes](#available-recipes)
- [Supported datasets](#supported-datasets)
- [Calls for contributions](#contributing)

## Installation
In order to install Asteroid, clone the repo and install it using 
pip or python :
```bash
git clone https://github.com/mpariente/asteroid
cd asteroid
# Install with pip in editable mode
pip install -e .
# Or, install with python in dev mode
python setup.py develop
```
Asteroid is also on PyPI, you can install the latest release 
with `pip install asteroid`


## Highlights
Here is a list of notebooks showing example usage of Asteroid's features.
- [Permutation invariant training wrapper `PITLossWrapper`][pitwrapper_nb]
- [Filterbank API][fb_nb]


## Running a recipe
Running the recipes requires additional packages in most cases, 
we recommend running :
```bash
# from asteroid/
pip install -r requirements.txt
```
Then choose the recipe you want to run and run it !
```bash
cd egs/wham/ConvTasNet
. ./run.sh
```
More information in [egs/README.md](./egs).

## Available recipes 
* [x] [ConvTasnet](./egs/wham/ConvTasNet) ([Luo et al.](https://arxiv.org/abs/1809.07454))
* [x] [Tasnet](./egs/whamr/TasNet) ([Luo et al.](https://arxiv.org/abs/1711.00541))
* [x] [Deep clustering](./egs/wsj0-mix/DeepClustering) ([Hershey et al.](https://arxiv.org/abs/1508.04306) and [Isik et al.](https://arxiv.org/abs/1607.02173))
* [ ] Chimera ++ ([Luo et al.](https://arxiv.org/abs/1611.06265) and [Wang et al.](https://ieeexplore.ieee.org/document/8462507))
* [x] [DualPathRNN](./egs/wham/DPRNN) ([Luo et al.](https://arxiv.org/abs/1910.06379))
* [ ] Two step learning (Coming) ([Tzinis et al.](https://arxiv.org/abs/1910.09804))
* [ ] Wavesplit (Coming) ([Zeghidour and Grangier](https://arxiv.org/abs/2002.08933))

## Supported datasets
* [x] [WSJ0-2mix](./egs/wsj0-mix) / WSJ03mix ([Hershey et al.](https://arxiv.org/abs/1508.04306))
* [x] [WHAM](./egs/wham) ([Wichern et al.](https://arxiv.org/abs/1907.01160))
* [x] [WHAMR](./egs/whamr) ([Maciejewski et al.](https://arxiv.org/abs/1910.10279))
* [x] [Microsoft DNS Challenge](./egs/dns_challenge) ([Chandan et al.](https://arxiv.org/abs/2001.08662))
* [x] [SMS_WSJ](./egs/sms_wsj) ([Drude et al.](https://arxiv.org/abs/1910.13934))
* [ ] MUSDB18 (Coming) ([Raffi et al.](https://hal.inria.fr/hal-02190845)) 
* [ ] FUSS (Coming) ([Wisdom et al.](https://zenodo.org/record/3694384#.XmUAM-lw3g4))

## Contributing
We are always looking to expand our coverage of the source separation 
and speech enhancement research, the following is a list of 
things we're missing. 
You want to contribute? This is a great place to start !
* FurcaNeXt ([Shi et al.](https://arxiv.org/abs/1902.04891))
* DeepCASA ([Liu and Want](https://arxiv.org/abs/1904.11148))
* VCTK Test sets from [Kadioglu et al.](https://arxiv.org/pdf/2002.08688.pdf)
* Interrupted and cascaded PIT ([Yang et al.](https://arxiv.org/abs/1910.12706))
* Consistency contraints ([Wisdom et al.](https://ieeexplore.ieee.org/abstract/document/8682783))
* Backpropagable STOI and PESQ.
* Parametrized filterbanks from [Tukuljac et al.](https://openreview.net/forum?id=HyewT1BKvr)
* End-to-End MISI ([Wang et al.](https://arxiv.org/abs/1804.10204))
* Modified MISI ([Wang et al.](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/3242.pdf))


Don't forget to read our [contributing guidelines](./CONTRIBUTING.md).

You can also open an issue or make a PR to add something we missed in this list.

## TensorBoard visualization
The default logger is TensorBoard in all the recipes. From the recipe folder, 
you can run the following to visualize the logs of all your runs. You can 
also compare different systems on the same dataset by running a similar command
from the dataset directiories.
```bash
# Launch tensorboard (default port is 6006)
tensorboard --logdir exp/ --port tf_port
```
If your launching tensorboard remotely, you should open an ssh tunnel
```bash
# Open port-forwarding connection. Add -Nf option not to open remote. 
ssh -L local_port:localhost:tf_port user@ip
```
Then open `http://localhost:local_port/`. If both ports are the same, you can 
click on the tensorboard URL given on the remote, it's just more practical.


## Guiding principles
* __Modularity.__ Building blocks are thought and designed to be seamlessly
plugged together. Filterbanks, encoders, maskers, decoders and losses are 
all common building blocks that can be combined in a 
flexible way to create new systems.  
* __Extensibility.__ Extending Asteroid with new features is simple.
Add a new filterbank, separator architecture, dataset or even recipe very 
easily.
* __Reproducibility.__ Recipes provide an easy way to reproduce 
results with data preparation, system design, training and evaluation in a 
single script. This is an essential tool for the community !


[comment]: <> (Badge)
[travis]: https://travis-ci.com/mpariente/asteroid
[travis-badge]: https://travis-ci.com/mpariente/asteroid.svg?branch=master
[codecov-badge]: https://codecov.io/gh/mpariente/asteroid/branch/master/graph/badge.svg
[codecov]: https://codecov.io/gh/mpariente/asteroid
[slack-badge]: https://img.shields.io/badge/slack-chat-green.svg?logo=slack
[slack-invite]: https://join.slack.com/t/asteroid-dev/shared_invite/zt-cn9y85t3-QNHXKD1Et7qoyzu1Ji5bcA

[comment]: <> (Notebooks)
[fb_nb]: https://github.com/mpariente/asteroid/blob/master/examples/Filterbank.ipynb
[pitwrapper_nb]: https://github.com/mpariente/asteroid/blob/master/examples/PITLossWrapper.ipynb

[comment]: <> (Others)
[issue]: https://github.com/mpariente/asteroid/issues/new
[pr]: https://github.com/mpariente/asteroid/compare