.. role:: hidden
    :class: hidden-section

Filterbank API
==============

Filterbank, Encoder and Decoder
-------------------------------
.. autoclass:: asteroid_filterbanks.Filterbank
   :members:
.. autoclass:: asteroid_filterbanks.Encoder
   :members:
   :show-inheritance:
.. autoclass:: asteroid_filterbanks.Decoder
   :members:
   :show-inheritance:
.. autoclass:: asteroid_filterbanks.make_enc_dec
   :members:
.. autoclass:: asteroid_filterbanks.get

Learnable filterbanks
---------------------

:hidden:`Free`
~~~~~~~~~~~~~~~~
.. automodule:: asteroid_filterbanks.free_fb
    :members:

:hidden:`Analytic Free`
~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: asteroid_filterbanks.analytic_free_fb
    :members:

:hidden:`Parameterized Sinc`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: asteroid_filterbanks.param_sinc_fb
    :members:

Fixed filterbanks
-----------------

:hidden:`STFT`
~~~~~~~~~~~~~~~~
.. automodule:: asteroid_filterbanks.stft_fb
    :members:

:hidden:`MelGram`
~~~~~~~~~~~~~~~~~
.. automodule:: asteroid_filterbanks.melgram_fb
    :members:

:hidden:`MPGT`
~~~~~~~~~~~~~~~~
.. autoclass:: asteroid_filterbanks.multiphase_gammatone_fb.MultiphaseGammatoneFB
    :members:

Transforms
----------

:hidden:`Griffin-Lim and MISI`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: asteroid_filterbanks.griffin_lim
   :members:

:hidden:`Complex transforms`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: asteroid_filterbanks.transforms
   :members:
