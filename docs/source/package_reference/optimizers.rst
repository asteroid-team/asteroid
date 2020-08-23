Optimizers & Schedulers
=======================

Optimizers
----------

Asteroid relies on ``torch_optimizer`` and ``torch`` for optimizers.
We provide a simple ``get`` method that retrieves optimizers from string,
which makes it easy to specify optimizers from the command line.

Here is a list of supported optimizers, retrievable from string:

- AccSGD
- AdaBound
- AdaMod
- DiffGrad
- Lamb
- NovoGrad
- PID
- QHAdam
- QHM
- RAdam
- SGDW
- Yogi
- Ranger
- RangerQH
- RangerVA
- Adam
- RMSprop
- SGD
- Adadelta
- Adagrad
- Adamax
- AdamW
- ASG

.. automodule:: asteroid.engine.optimizers
   :members:


Schedulers
----------

Asteroid provides step-wise learning schedulers, integrable to
``pytorch-lightning`` via ``System``.

.. automodule:::: asteroid.engine.schedulers
   :members:
