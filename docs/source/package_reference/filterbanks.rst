.. role:: hidden
    :class: hidden-section

Filterbank API
==============

Filterbank, Encoder and Decoder
-----------------
.. autoclass:: asteroid.filterbanks.Filterbank
   :members:
.. autoclass:: asteroid.filterbanks.Encoder
   :members:
   :show-inheritance:
.. autoclass:: asteroid.filterbanks.Decoder
   :members:
   :show-inheritance:
.. autoclass:: asteroid.filterbanks.make_enc_dec
   :members:
.. autoclass:: asteroid.filterbanks.get

Learnable filterbanks
---------------------

:hidden:`Free`
~~~~~~~~~~~~~~~~
.. automodule:: asteroid.filterbanks.free_fb
    :members:

:hidden:`Analytic Free`
~~~~~~~~~~~~~~~~
.. automodule:: asteroid.filterbanks.analytic_free_fb
    :members:

:hidden:`Parameterized Sinc`
~~~~~~~~~~~~~~~~
.. automodule:: asteroid.filterbanks.param_sinc_fb
    :members:

Fixed filterbanks
-----------------

:hidden:`STFT`
~~~~~~~~~~~~~~~~
.. automodule:: asteroid.filterbanks.stft_fb
    :members:

:hidden:`MP-GTFB`
~~~~~~~~~~~~~~~~
.. automodule:: asteroid.filterbanks.multiphase_gammatone_fb
    :members:

Transforms
----------

:hidden:`Griffin-Lim and MISI`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: asteroid.filterbanks.griffin_lim
   :members:

:hidden:`Complex transforms`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: asteroid.filterbanks.transforms
   :members:

