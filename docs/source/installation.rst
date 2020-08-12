Installation
============

By following the instructions below, first install PyTorch and then
Asteroid (using either pip/dev install). We recommend the development
installation for users likely to modify the source code.

CUDA and PyTorch
****************

Asteroid is based on PyTorch.
To run Asteroid on GPU, you will need a CUDA-enabled PyTorch installation.
Visit this site for the instructions: https://pytorch.org/get-started/locally/.

Pip
***

Asteroid is regularly updated on PyPI, install the latest stable version with::

    pip install asteroid


Development installation
************************

For development installation, you can fork/clone the GitHub repo and locally install it with pip::

    git clone https://github.com/mpariente/asteroid
    cd asteroid
    pip install -e .

This is an editable install (``-e`` flag), it means that source code changes (or branch switching) are
automatically taken into account when importing asteroid.
