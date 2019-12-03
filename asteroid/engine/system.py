"""
Proposed base class to interface with pytorch-lightning.
@author : Manuel Pariente, Inria-Nancy
"""

import torch
import pytorch_lightning as pl


class System(pl.LightningModule):
    """ Base class for deep learning system.
    Subclass of pytorch_lightning.LightningModule.
    Contains a model, an optimizer, a loss_class and training and validation
    loaders and learning rate scheduler.

    Args:
        model: nn.Module instance.
        optimizer: torch.optim.Optimizer instance, or list of.
        loss_class: Class with `compute` method. (More doc to come)
        train_loader: torch.utils.data.DataLoader instance.
        val_loader: torch.utils.data.DataLoader instance.
        scheduler: torch.optim._LRScheduler instance, or list of.
        config: Anything to be saved with the checkpoints during training.
            The config dictionary to re-instantiate the run for example.
    Methods:
        # From System
        common_step : Common step between training and validation.
        unpack_data : Unpacks batch from loaders.

        # Overwritten from pl.LightningModule
        forward: self.model.forward (Unused butrequired by PL)
        training_step : Training step (doesn't include backward and GD)
        validation_step : Validation step (forward and compute loss)
        validation_end : Aggregates results and outputs logs.
        configure_optimizer: return self.optimizer, self.scheduler by default.
        train_dataloader: return self.train_loader
        val_dataloader: return self.val_loader
        on_checkpoint: adds self.config in checkpoint['training_config']

    By default, `training_step` (used by pytorch-lightning in the training
    loop) and `validation_step` (used for the validation loop) share
    `common_step`. If you want different behavior for the training loop and
    the validation loop, overwrite both `training_step` and `validation_step`
    instead.
    """
    def __init__(self, model, optimizer, loss_class, train_loader,
                 val_loader=None, scheduler=None, config=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_class = loss_class
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.config = config

    def forward(self, *args, **kwargs):
        """ Required by PL.

        Returns:
            self.model.forward
        """
        return self.model.forward(*args, **kwargs)

    def common_step(self, batch, batch_nb):
        """ Common forward step between training and validation.
        The function of this method is to unpack the data given by the loader,
        forward the batch through the model and compute the loss.
        Args:
            batch: the object returned by the loader (a list of torch.Tensor
                in most cases) but can be something else.
            batch_nb: The number of the batch in the epoch.
        Returns:
            The loss value on this batch.

        Note : this is typically the method to overwrite when subclassing
        `System`. If the training and validation steps are different (except
        for loss.backward() and optimzer.step()), then overwrite
        `training_step` and `validation_step` instead.
        """
        inputs, targets, infos = self.unpack_data(batch)
        est_targets = self.model(inputs)
        loss = self.loss_class.compute(targets, est_targets, infos=infos)
        return loss

    def unpack_data(self, data):
        """
         Unpack data given by the DataLoader
        Args:
            data: list of 2 or 3 elements.
                [model_inputs, training_targets, additional infos] or
                [model_inputs, training_targets]
        Returns:
              model_inputs, training_targets, additional infos
        """
        if len(data) == 2:
            inputs, targets = data
            infos = dict()
        elif len(data) == 3:
            inputs, targets, infos = data
        else:
            raise ValueError('Expected DataLoader output to have '
                             '2 or 3 elements. Received '
                             '{} elements'.format(len(data)))
        return inputs, targets, infos

    def training_step(self, batch, batch_nb):
        """ Pass data through the model and compute the loss. """
        loss = self.common_step(batch, batch_nb)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        """ Need to overwrite PL validation_step to do validation. """
        loss = self.common_step(batch, batch_nb)
        return {'val_loss': loss}

    def validation_end(self, outputs):
        """ How to aggregate outputs of `validation_step` for logging."""
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs,
                'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        """ Required by pytorch-lightning. """
        if self.scheduler is not None:
            return self.optimizer, self.scheduler
        return self.optimizer

    @pl.data_loader
    def train_dataloader(self):
        return self.train_loader

    @pl.data_loader
    def val_dataloader(self):
        return self.val_loader

    def on_save_checkpoint(self, checkpoint):
        """ Overwrite if you want to save more things in the checkpoint."""
        checkpoint['training_config'] = self.config
        return checkpoint

    def on_batch_start(self, batch):
        """ Overwrite if needed. Called by pytorch-lightning"""
        pass

    def on_batch_end(self):
        """ Overwrite if needed. Called by pytorch-lightning"""
        pass

    def on_epoch_start(self):
        """ Overwrite if needed. Called by pytorch-lightning"""
        pass

    def on_epoch_end(self):
        """ Overwrite if needed. Called by pytorch-lightning"""
        pass

    @pl.data_loader
    def tng_dataloader(self):
        """ Deprecated."""
        pass
