.. role:: hidden
    :class: hidden-section

Losses & Metrics
================

.. automodule:: asteroid.losses
.. currentmodule:: asteroid.losses


Permutation invariant training (PIT) made easy
----------------------------------------------

Asteroid supports regular Permutation Invariant Training (PIT), it's extension
using Sinkhorn algorithm (SinkPIT) as well as Mixture Invariant Training (MixIT).


:hidden:`PIT`
~~~~~~~~~~~~~~~~
.. automodule:: asteroid.losses.pit_wrapper
   :members:

:hidden:`MixIT`
~~~~~~~~~~~~~~~~
.. automodule:: asteroid.losses.mixit_wrapper
   :members:

:hidden:`SinkPIT`
~~~~~~~~~~~~~~~~~
.. automodule:: asteroid.losses.sinkpit_wrapper
   :members:


Available loss functions
------------------------

``PITLossWrapper`` supports three types of loss function. For "easy" losses,
we implement the three types (pairwise point, single-source loss and multi-source loss).
For others, we only implement the single-source loss which can be aggregated
into both PIT and nonPIT training.

:hidden:`MSE`
~~~~~~~~~~~~~~~~

.. autofunction:: asteroid.losses.mse.PairwiseMSE
.. autofunction:: asteroid.losses.mse.SingleSrcMSE
.. autofunction:: asteroid.losses.mse.MultiSrcMSE


:hidden:`SDR`
~~~~~~~~~~~~~~~~

.. autofunction:: asteroid.losses.sdr.PairwiseNegSDR
.. autofunction:: asteroid.losses.sdr.SingleSrcNegSDR
.. autofunction:: asteroid.losses.sdr.MultiSrcNegSDR


:hidden:`PMSQE`
~~~~~~~~~~~~~~~~

.. autofunction:: asteroid.losses.pmsqe.SingleSrcPMSQE

:hidden:`STOI`
~~~~~~~~~~~~~~~~
.. autofunction:: asteroid.losses.stoi.NegSTOILoss


:hidden:`MultiScale Spectral Loss`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: asteroid.losses.multi_scale_spectral.SingleSrcMultiScaleSpectral

:hidden:`Deep clustering (Affinity) loss`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: asteroid.losses.cluster.deep_clustering_loss


Computing metrics
-----------------

.. autofunction:: asteroid.metrics.get_metrics
