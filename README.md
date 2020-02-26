# Asteroid : Audio Source Separation on steroids
[![Build Status](https://travis-ci.com/mpariente/AsSteroid.svg?branch=master)](https://travis-ci.com/mpariente/AsSteroid)
[![codecov](https://codecov.io/gh/mpariente/AsSteroid/branch/master/graph/badge.svg)](https://codecov.io/gh/mpariente/AsSteroid)

[![Slack](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://join.slack.com/t/asteroid-dev/shared_invite/enQtOTM4NjEzOTI0MzQzLWMyODBmZjRiODAyOGZmNjQ0ZjVjZjM3NGM0NDIyOTc4ZjEyYjc0ZmI1NDI5N2I2YzE5OWU2ZGE1NmEyMjBlYTQ)

Asteroid is a Pytorch-based source separation and speech enhancement API 
that enables fast experimentation on common datasets. 
It comes with a source code written to support a large range of architectures 
and a set of recipes to reproduce some papers.  
Asteroid is intended to be a __community-based project__ 
so hop on and help us !

## Table of contents
- [Installation](https://github.com/mpariente/asteroid#installation)
- [Tutorials](https://github.com/mpariente/asteroid#highlights)
- [Running a recipe](https://github.com/mpariente/asteroid#running-a-recipe)
- [Available recipes](https://github.com/mpariente/asteroid#recipes)
- [Supported datasets](https://github.com/mpariente/asteroid#supported-datasets)
- [Our guiding principles](https://github.com/mpariente/asteroid#guiding-principles)

## Installation
In order to install Asteroid, clone the repo and install it using pip or python :
```bash
git clone https://github.com/mpariente/asteroid
cd asteroid
# Install with pip (in editable mode)
pip install -e .
# Install with python
python setup.py install
```

## Highlights
Few notebooks showing example usage of `asteroid`'s features.
- [Permutation invariant training wrapper `PITLossWrapper`
.](https://github.com/mpariente/asteroid/blob/master/examples/PITLossWrapper.ipynb)
- [Filterbank API](https://github.com/mpariente/asteroid/blob/master/examples/Filterbank.ipynb)


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
./run.sh
```
More information in [egs/README.md](https://github.com/mpariente/asteroid/tree/master/egs).

## Recipes 
* [x] ConvTasnet ([Luo et al.](https://arxiv.org/abs/1809.07454))
* [ ] Tasnet ([Luo et al.](https://arxiv.org/abs/1711.00541))
* [x] Deep clustering ([Hershey et al.](https://arxiv.org/abs/1508.04306) and [Isik et al.](https://arxiv.org/abs/1607.02173))
* [x] Chimera ++ ([Luo et al.](https://arxiv.org/abs/1611.06265) and [Wang et al.](https://ieeexplore.ieee.org/document/8462507))
* [ ] FurcaNeXt ([Shi et al.](https://arxiv.org/abs/1902.04891))
* [x] DualPathRNN ([Luo et al.](https://arxiv.org/abs/1910.06379))
* [ ] Two step learning ([Tzinis et al.](https://arxiv.org/abs/1910.09804))

## Supported datasets

* [x] WSJ0-2mix / WSJ03mix ([Hershey et al.](https://arxiv.org/abs/1508.04306))
* [x] WHAM ([Wichern et al.](https://arxiv.org/abs/1907.01160))
* [ ] WHAMR ([Maciejewski et al.](https://arxiv.org/abs/1910.10279))
* [x] Microsoft DNS Challenge ([Chandan et al.](https://arxiv.org/abs/2001.08662))
* [x] SMS_WSJ ([Drude et al.](https://arxiv.org/abs/1910.13934))

## Writing your own recipe

## Contributing
See our [contributing guidelines](https://github.com/mpariente/asteroid/blob/master/CONTRIBUTING.md).


## Building the docs
To build the docs, you'll need [Sphinx](https://www.sphinx-doc.org/en/master/), 
a theme and some other package
```bash
# Start by installing the required packages
cd docs/
pip install -r requirements.txt
# Build the docs
make html
# View it ! (Change firefox by your favorite browser)
firefox build/html/index.html
```
If you rebuild the docs, don't forget to run `make clean` before it.  

You can add this to your `.bashrc`, source it and run `run_docs` 
from the `docs/` folder
```bash
alias run_docs='make clean; make html; firefox build/html/index.html'
```

### Codebase structure
```
├── asteroid                 # Python package / Source code
│   ├── data                 # Data classes, DalatLoaders maker.
│   ├── engine               # Training classes : losses, optimizers and trainer.
│   ├── filterbanks          # Common filterbanks and related classes.
│   ├── masknn               # Separation building blocks and architectures.
│   └── utils.py
├── examples                 # Simple asteroid examples 
└── egs                      # Recipes for all datasets and systems.
│   ├── wham                 # Recipes for one dataset (WHAM) 
│   │   ├── ConvTasNet       # ConvTasnet systme on the WHAM dataset.
│   │   │   └── ...          # Recipe's structure. See egs/README.md for more info
│   │   ├── Your recipe      # More recipes on the same dataset (Including yours)
│   │   ├── ...
│   │   └── DualPathRNN
│   └── Your dataset         # More datasets (Including yours)

```

## Remote TensorBoard visualization
```bash
# Launch tensorboard remotely (default port is 6006)
tensorboard --logdir exp/tmp/lightning_logs/ --port tf_port

# Open port-forwarding connection. Add -Nf option not to open remote. 
ssh -L local_port:localhost:tf_port user@ip
```
Then open `http://localhost:local_port/`.


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
same script. This is an essential tool for the community !
