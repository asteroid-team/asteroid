.. role:: hidden
    :class: hidden-section

.. currentmodule:: asteroid.masknn.blocks

DNN building blocks
===================

Mask estimators
---------------

:hidden:`Ready-to-use`
~~~~~~~~~~~~~~~~
.. autofunction:: TDConvNet
.. autofunction:: DPRNN


:hidden:`Layers`
~~~~~~~~~~~~~~~~

.. autofunction:: Conv1DBlock
.. autofunction:: SingleRNN
.. autofunction:: DPRNNBlock

Normalization layers
--------------------

.. currentmodule:: asteroid.masknn.norms
.. autofunction:: GlobLN
.. autofunction:: ChanLN
.. autofunction:: CumLN
