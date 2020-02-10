# Asteroid : Audio Source Separation on steroids
[![Build Status](https://travis-ci.com/mpariente/AsSteroid.svg?branch=master)](https://travis-ci.com/mpariente/AsSteroid)
[![codecov](https://codecov.io/gh/mpariente/AsSteroid/branch/master/graph/badge.svg)](https://codecov.io/gh/mpariente/AsSteroid)

[![Slack](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://join.slack.com/t/asteroid-dev/shared_invite/enQtOTM4NjEzOTI0MzQzLWMyODBmZjRiODAyOGZmNjQ0ZjVjZjM3NGM0NDIyOTc4ZjEyYjc0ZmI1NDI5N2I2YzE5OWU2ZGE1NmEyMjBlYTQ)
### :construction: :warning: Under development :warning: :construction:
Asteroid is a Pytorch-based source separation and speech enhancement API 
that enables fast experimentation on common datasets. 
It comes with a source code written to support a large range of architectures 
and a set of recipes to reproduce some papers.  
Asteroid is intended to be a __community-based project__ 
so hop on and help us !

## Guiding principles
* __User friendliness.__ Asteroid's API offers simple solutions for most 
common use cases.
* __Modularity.__ Building blocks are thought and designed to be seamlessly
plugged together. Filterbanks, encoders, maskers, decoders and losses are 
all common building blocks that can be combined in a 
flexible way to create new systems.  
* __Extensibility.__ Extending Asteroid with new features is simple.
Add a new filterbank, separator, architecture, dataset or even recipe very 
easily.
* __Reproducibility.__ Recipes provide an easy way to reproduce 
results with data preparation, training and evaluation in a same script. 


## Highlights
- [Permutation invariant training wrapper `PITLossWrapper`
.](https://github.com/mpariente/AsSteroid/blob/master/examples/PITLossWrapper.ipynb)
- [Filterbank API](https://github.com/mpariente/AsSteroid/blob/master/examples/Filterbank.ipynb)

## Installation
In order to install Asteroid, clone the repo and install it using pip or python :
```bash
git clone https://github.com/mpariente/AsSteroid
cd AsSteroid
# Install with pip (in editable mode)
pip install -e .
# Install with python
python setup.py install
```

Running the recipes requires additional packages in most cases, 
we recommend running :
```bash
pip install numpy Cython scipy
# from AsSteroid/
pip install -r requirements.txt
```

## Running a recipe
```bash
cd egs/wham/ConvTasNet
./run.sh
```
More information in [egs/README.md](https://github.com/mpariente/AsSteroid/tree/master/egs/README.md).

## Recipes 
* [ ] ConvTasnet ([Luo et al.](https://arxiv.org/abs/1809.07454))
* [ ] Tasnet ([Luo et al.](https://arxiv.org/abs/1711.00541))
* [ ] Deep clustering ([Hershey et al.](https://arxiv.org/abs/1508.04306) and [Isik et al.](https://arxiv.org/abs/1607.02173))
* [ ] Chimera ++ (for ) ([Luo et al.](https://arxiv.org/abs/1611.06265) and [Wang et al.](https://ieeexplore.ieee.org/document/8462507))
* [ ] FurcaNeXt ([Shi et al.](https://arxiv.org/abs/1902.04891))
* [ ] DualPathRNN ([Luo et al.](https://arxiv.org/abs/1910.06379))
* [ ] Two step learning ([Tzinis et al.](https://arxiv.org/abs/1910.09804))

## Writing your own recipe

## Contributing
See our [contributing guidelines](https://github.com/mpariente/AsSteroid/blob/master/CONTRIBUTING.md).

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
for the `docs/` folder
```bash
alias run_docs='make clean; make html; firefox build/html/index.html'
```
## Why Asteroid ? 
Audio source separation and speech enhancement are fast evolving fields with 
a growing number of papers submitted to conferences each year. While datasets 
such as [wsj0-{2, 3}mix](http://www.merl.com/demos/deep-clustering),
[WHAM](http://wham.whisper.ai/) or 
[MS-SNSD](https://github.com/microsoft/MS-SNSD) are being shared, there has 
been little effort to create common codebases for development and evaluation 
of source separation and speech enhancement algorithms. Here is one !

## Remote TensorBoard visualization
```bash
# Launch tensorboard remotely (default port is 6006)
tensorboard --logdir exp/tmp/lightning_logs/ --port tf_port

# Open port-forwarding connection. Add -Nf option not to open remote. 
ssh -L local_port:localhost:tf_port user@ip
```
Then open `http://localhost:local_port/`.
