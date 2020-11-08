.. asteroid documentation master file, created by
   sphinx-quickstart on Sat Dec  7 14:48:37 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Asteroid: Audio source separation on Steroids
=============================================

.. image:: ./_static/images/asteroid_logo_dark.png

Asteroid is a Pytorch-based audio source separation toolkit that enables fast
experimentation on common datasets. It comes with a source code that supports a
large range of datasets and architectures, and a set of recipes to reproduce some important papers.

.. toctree::
   :maxdepth: 1
   :caption: Start here

   why_use_asteroid
   installation

.. toctree::
   :maxdepth: 1
   :caption: Notebooks and Tutorials

   Getting started with Asteroid <http://colab.research.google.com/github/mpariente/asteroid/blob/master/notebooks/00_AsteroidGettingStarted.ipynb>
   Introduction and Overview <http://colab.research.google.com/github/mpariente/asteroid/blob/master/notebooks/01_AsteroidOverview.ipynb>
   Understanding the Filterbank API <http://colab.research.google.com/github/mpariente/asteroid/blob/master/notebooks/02_Filterbank.ipynb>
   Our PITLossWrapper explained <http://colab.research.google.com/github/mpariente/asteroid/blob/master/notebooks/03_PITLossWrapper.ipynb>
   Processing large wav files <http://colab.research.google.com/github/mpariente/asteroid/blob/master/notebooks/04_ProcessLargeAudioFiles.ipynb>
   Community: Numpy vs. Asteroid STFT <https://colab.research.google.com/drive/1BDNQZBJCDcwQhSguf3XBE7ff2KXhWu_j>

.. toctree::
   :maxdepth: 1
   :caption: Asteroid

   readmes/egs_README.md
   supported_datasets
   training_and_evaluation
   readmes/pretrained_models.md
   faq

.. toctree::
   :maxdepth: 1
   :caption: Package reference

   package_reference/data
   package_reference/filterbanks
   package_reference/blocks
   package_reference/models
   package_reference/losses
   package_reference/system
   package_reference/optimizers
   package_reference/dsp
   package_reference/utils
   package_reference/cli

.. toctree::
   :maxdepth: 1
   :caption: Community

   contribution_guide
   readmes/CONTRIBUTING.md


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
