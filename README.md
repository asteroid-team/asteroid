<div align="center">
<img src="docs/source/_static/images/asteroid_logo_dark.png" width="50%">

**The PyTorch-based audio source separation toolkit for researchers.**

[![PyPI Status](https://badge.fury.io/py/asteroid.svg)](https://badge.fury.io/py/asteroid)
[![Build Status](https://github.com/asteroid-team/asteroid/workflows/CI/badge.svg)](https://github.com/asteroid-team/asteroid/actions?query=workflow%3ACI+branch%3Amaster+event%3Apush)
[![codecov][codecov-badge]][codecov]
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://img.shields.io/badge/docs-0.4.5-blue)](https://asteroid.readthedocs.io/en/v0.4.5/)
[![Latest Docs Status](https://github.com/asteroid-team/asteroid/workflows/Latest%20docs/badge.svg)](https://asteroid-team.github.io/asteroid/)


[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/asteroid-team/asteroid/pulls)
[![Python Versions](https://img.shields.io/pypi/pyversions/asteroid.svg)](https://pypi.org/project/asteroid/)
[![PyPI Status](https://pepy.tech/badge/asteroid)](https://pepy.tech/project/asteroid)
[![Slack][slack-badge]][slack-invite]

</div>

--------------------------------------------------------------------------------


Asteroid is a Pytorch-based audio source separation toolkit
that enables fast experimentation on common datasets.
It comes with a source code that supports a large range
of datasets and architectures, and a set of
 recipes to reproduce some important papers.


### You use Asteroid or you want to?
Please, if you have found a bug, [open an issue][issue],
if you solved it, [open a pull request][pr]!
Same goes for new features, tell us what you want or help us building it!
Don't hesitate to [join the slack][slack-invite]
and ask questions / suggest new features there as well!
Asteroid is intended to be a __community-based project__
so hop on and help us!
## Contents
- [Installation](#installation)
- [Tutorials](#tutorials)
- [Running a recipe](#running-a-recipe)
- [Available recipes](#available-recipes)
- [Supported datasets](#supported-datasets)
- [Pretrained models](#pretrained-models)
- [Calls for contributions](#contributing)
- [Citing us](#citing)

## Installation
([↑up to contents](#contents))
To install Asteroid, clone the repo and install it using
conda, pip or python :
```bash
# First clone and enter the repo
git clone https://github.com/asteroid-team/asteroid
cd asteroid
```

- With `pip`
```bash
# Install with pip in editable mode
pip install -e .
# Or, install with python in dev mode
# python setup.py develop
```
- With conda (if you don't already have conda, see [here][miniconda].)
```bash
conda env create -f environment.yml
conda activate asteroid
```

- Asteroid is also on PyPI, you can install the latest release with
```bash
pip install asteroid
```

## Tutorials
([↑up to contents](#contents))
Here is a list of notebooks showing example usage of Asteroid's features.
- [Getting started with Asteroid](./notebooks/00_GettingStarted.ipynb)
- [Introduction and Overview](./notebooks/01_APIOverview.ipynb)
- [Filterbank API](./notebooks/02_Filterbank.ipynb)
- [Permutation invariant training wrapper `PITLossWrapper`](./notebooks/03_PITLossWrapper.ipynb)
- [Process large wav files](./notebooks/04_ProcessLargeAudioFiles.ipynb)


## Running a recipe
([↑up to contents](#contents))
Running the recipes requires additional packages in most cases,
we recommend running :
```bash
# from asteroid/
pip install -r requirements.txt
```
Then choose the recipe you want to run and run it!
```bash
cd egs/wham/ConvTasNet
. ./run.sh
```
More information in [egs/README.md](./egs).

## Available recipes
([↑up to contents](#contents))
* [x] [ConvTasnet](./egs/wham/ConvTasNet) ([Luo et al.](https://arxiv.org/abs/1809.07454))
* [x] [Tasnet](./egs/whamr/TasNet) ([Luo et al.](https://arxiv.org/abs/1711.00541))
* [x] [Deep clustering](./egs/wsj0-mix/DeepClustering) ([Hershey et al.](https://arxiv.org/abs/1508.04306) and [Isik et al.](https://arxiv.org/abs/1607.02173))
* [x] [Chimera ++](./egs/wsj0-mix/DeepClustering) ([Luo et al.](https://arxiv.org/abs/1611.06265) and [Wang et al.](https://ieeexplore.ieee.org/document/8462507))
* [x] [DualPathRNN](./egs/wham/DPRNN) ([Luo et al.](https://arxiv.org/abs/1910.06379))
* [x] [Two step learning](./egs/wham/TwoStep)([Tzinis et al.](https://arxiv.org/abs/1910.09804))
* [x] [SudoRMRFNet](./asteroid/models/sudormrf.py) ([Tzinis et al.](https://arxiv.org/abs/2007.06833))
* [x] [DPTNet](./asteroid/models/dptnet.py) ([Chen et al.](https://arxiv.org/abs/2007.13975))
* [x] [DCCRNet](./asteroid/models/dccrnet.py) ([Hu et al.](https://arxiv.org/abs/2008.00264))
* [x] [DCUNet](./asteroid/models/dcunet.py) ([Choi et al.](https://arxiv.org/abs/1903.03107))
* [ ] Open-Unmix (coming) ([Stöter et al.](https://sigsep.github.io/open-unmix/))
* [ ] Wavesplit (coming) ([Zeghidour et al.](https://arxiv.org/abs/2002.08933))

## Supported datasets
([↑up to contents](#contents))
* [x] [WSJ0-2mix](./egs/wsj0-mix) / WSJ03mix ([Hershey et al.](https://arxiv.org/abs/1508.04306))
* [x] [WHAM](./egs/wham) ([Wichern et al.](https://arxiv.org/abs/1907.01160))
* [x] [WHAMR](./egs/whamr) ([Maciejewski et al.](https://arxiv.org/abs/1910.10279))
* [x] [LibriMix](./egs/librimix) ([Cosentino et al.](https://arxiv.org/abs/2005.11262))
* [x] [Microsoft DNS Challenge](./egs/dns_challenge) ([Chandan et al.](https://arxiv.org/abs/2001.08662))
* [x] [SMS_WSJ](./egs/sms_wsj) ([Drude et al.](https://arxiv.org/abs/1910.13934))
* [x] [MUSDB18](./asteroid/data/musdb18_dataset.py) ([Raffi et al.](https://hal.inria.fr/hal-02190845))
* [x] [FUSS](./asteroid/data/fuss_dataset.py) ([Wisdom et al.](https://zenodo.org/record/3694384#.XmUAM-lw3g4))
* [x] [AVSpeech](./asteroid/data/avspeech_dataset.py) ([Ephrat et al.](https://arxiv.org/abs/1804.03619))
* [x] [Kinect-WSJ](./asteroid/data/kinect_wsj.py) ([Sivasankaran et al.](https://github.com/sunits/Reverberated_WSJ_2MIX))

## Pretrained models
([↑up to contents](#contents))
See [here](./docs/source/readmes/pretrained_models.md)

## Contributing
([↑up to contents](#contents))
We are always looking to expand our coverage of the source separation
and speech enhancement research, the following is a list of
things we're missing.
You want to contribute? This is a great place to start!
* Wavesplit ([Zeghidour and Grangier](https://arxiv.org/abs/2002.08933))
* FurcaNeXt ([Shi et al.](https://arxiv.org/abs/1902.04891))
* DeepCASA ([Liu and Want](https://arxiv.org/abs/1904.11148))
* VCTK Test sets from [Kadioglu et al.](https://arxiv.org/pdf/2002.08688.pdf)
* Interrupted and cascaded PIT ([Yang et al.](https://arxiv.org/abs/1910.12706))
* ~Consistency contraints ([Wisdom et al.](https://ieeexplore.ieee.org/abstract/document/8682783))~
* ~Backpropagable STOI and PESQ.~
* Parametrized filterbanks from [Tukuljac et al.](https://openreview.net/forum?id=HyewT1BKvr)
* ~End-to-End MISI ([Wang et al.](https://arxiv.org/abs/1804.10204))~


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
([↑up to contents](#contents))
* __Modularity.__ Building blocks are thought and designed to be seamlessly
plugged together. Filterbanks, encoders, maskers, decoders and losses are
all common building blocks that can be combined in a
flexible way to create new systems.
* __Extensibility.__ Extending Asteroid with new features is simple.
Add a new filterbank, separator architecture, dataset or even recipe very
easily.
* __Reproducibility.__ Recipes provide an easy way to reproduce
results with data preparation, system design, training and evaluation in a
single script. This is an essential tool for the community!

## Citing Asteroid
([↑up to contents](#contents))
If you loved using Asteroid and you want to cite us, use this :
```BibTex
@inproceedings{Pariente2020Asteroid,
    title={Asteroid: the {PyTorch}-based audio source separation toolkit for researchers},
    author={Manuel Pariente and Samuele Cornell and Joris Cosentino and Sunit Sivasankaran and
            Efthymios Tzinis and Jens Heitkaemper and Michel Olvera and Fabian-Robert Stöter and
            Mathieu Hu and Juan M. Martín-Doñas and David Ditter and Ariel Frank and Antoine Deleforge
            and Emmanuel Vincent},
    year={2020},
    booktitle={Proc. Interspeech},
}
```

[comment]: <> (Badge)
[miniconda]: https://conda.io/miniconda.html
[codecov-badge]: https://codecov.io/gh/asteroid-team/asteroid/branch/master/graph/badge.svg
[codecov]: https://codecov.io/gh/asteroid-team/asteroid
[slack-badge]: https://img.shields.io/badge/slack-chat-green.svg?logo=slack
[slack-invite]: https://join.slack.com/t/asteroid-dev/shared_invite/zt-cn9y85t3-QNHXKD1Et7qoyzu1Ji5bcA

[comment]: <> (Others)
[issue]: https://github.com/asteroid-team/asteroid/issues/new
[pr]: https://github.com/asteroid-team/asteroid/compare
