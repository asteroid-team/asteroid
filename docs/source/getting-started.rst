Getting started
===============

Asteroid is a PyTorch-based audio source separation toolkit.

The main goals of Asteroid are:

- Gather a wider **community** around audio source separation by lowering the barriers to entry.
- Provide PyTorch ``Dataset`` for **common datasets**.
- **Promote reproducibility** by replicating important research papers.
- Automatize most engineering and **make way for research**.
- Simplify **model sharing** to reduce compute costs and carbon footprint.


Installation
------------

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

    # Running unit tests
    pip install pytest
    python -m pytest tests/  # Ensure the right pytest

    # Build the docs
    cd docs/
    make html
    firefox build/html/index.html  # Or another browser


This is an editable install (``-e`` flag), it means that source code changes (or branch switching) are
automatically taken into account when importing asteroid.
