Training and Evaluation
=======================

Training and evaluation are the two essential parts of the recipes.
For training, we offer a thin wrapper around
`PyTorchLightning <https://github.com/PyTorchLightning/pytorch-lightning>`_ that
seamlessly enables distributed training, experiment logging and more,
without sacrificing flexibility.
For evaluation we released ``pb_bss_eval`` on PyPI, which is the evaluation
part of `pb_bss <https://github.com/fgnt/pb_bss>`_. All the credit goes to the
original authors from the Paderborn University.

Training with PyTorchLightning
------------------------------
First, have a look `here <https://pytorch-lightning.readthedocs.io/en/latest/introduction_guide.html>`_
for an overview of PyTorchLightning.
As you saw, the ``LightningModule`` is a central class of PyTorchLightning
where a large part of the research-related logic lives.
Instead of subclassing it everytime, we use ``System``, a thin wrapper
that separately gathers the essential parts of every deep learning project:

1. A model
2. Optimizer
3. Loss function
4. Train/val data

.. code-block:: python

	class System(pl.LightningModule):
		def __init__(self, model, optimizer, loss_func, train_loader, val_loader):
			...

		def common_step(self, batch):
		 """ common_step is the method that'll be called at both train/val time. """
			inputs, targets = batch
			est_targets = self(inputs)
			loss = self.loss_func(est_targets, targets)
			return loss

Only overwriting ``common_step`` will often be enough to obtain a desired
behavior, while avoiding boilerplate code.
Then, we can use the native PyTorchLightning ``Trainer`` to train the models.

Evaluation
----------

Asteroid's function ``compute_metrics`` that calls ``pb_bss_eval``
is used to compute the following common source separation metrics:

- SDR / SIR / SAR
- STOI
- PESQ
- SI-SDR


