.. asteroid documentation master file, created by
   sphinx-quickstart on Sat Dec  7 14:48:37 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Asteroid: Audio source separation on Steroids
=============================================

.. image:: ./_static/images/asteroid_logo_dark.png

Asteroid is a Pytorch-based audio source separation toolkit that enables fast
experimentation on common datasets. It comes with a source code thats supports a
large range of datasets and architectures, and a set of recipes to reproduce some important papers.

.. toctree::
   :maxdepth: 1
   :caption: Start here

   why_use_asteroid
   installation

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

   filterbanks
   losses
   masknn
   running_a_recipe

.. toctree::
   :maxdepth: 1
   :caption: Extending asteroid

   writing_a_new_loss
   writing_a_new_filterbank



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
