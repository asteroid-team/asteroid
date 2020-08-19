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
.. autofunction:: get

Activation layers
-----------------

.. currentmodule:: asteroid.masknn.activations
.. autofunction:: get
.. autofunction:: linear
.. autofunction:: relu
.. autofunction:: prelu
.. autofunction:: leaky_relu
.. autofunction:: sigmoid
.. autofunction:: softmax(dim=Non
.. autofunction:: tanh
.. autofunction:: gelu
.. autofunction:: swish
