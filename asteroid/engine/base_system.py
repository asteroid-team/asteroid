import torch
import pytorch_lightning as pl


class BaseSystem(pl.LightningModule):
    """ Base class for deep learning system.
    Subclass of pytorch_lightning.LightningModule.
    Contains a model, an optimizer, a loss_class and loaders.
    Needs to be subclassed most of the time to perform training.

    Args:
        model: nn.Module instance.
        optimizer: optim.Optimizer instance.
        loss_class: Class with `compute` method. (More doc to come)
        train_loader: torch.utils.data.DataLoader instance.
        val_loader: torch.utils.data.DataLoader instance.

    When subclassed, at least the `common_step` method needs to be
    implemented. `common_step` is the method which forward the data through
    the model (doesn't backward the loss).
    By default, `training_step` (used by pytorch-lightning in the training
    loop) and `validation_step` (used for the validation loop) share
    `common_step`. If you want different behavior for the training loop and
    the validation loop, overwrite both `training_step` and `validation_step`
    instead.
    In the simplest case, you can write
    >>> class System(BaseSystem):
    >>>     def common_step(self, batch, batch_nb):
    >>>         inputs, targets = batch
    >>>         est_targets = self.model(inputs)
    >>>         loss = self.loss_class.compute(targets, est_targets)
    >>>         return loss
    """
    def __init__(self, model, optimizer, loss_class, train_loader,
                 val_loader=None):
        super().__init__()
        self.model = model
        # Handle optimizer as a string ?
        self.optimizer = optimizer
        self.loss_class = loss_class
        self.train_loader = train_loader
        self.val_loader = val_loader

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

        In the most simple case, the common step would look like.
        >>> inputs, targets = batch
        >>> est_targets = self.model(inputs)
        >>> loss = self.loss_class.compute(targets, est_targets)
        >>> return loss
        """
        raise NotImplementedError

    def training_step(self, batch, batch_nb):
        """ Pass data through the model and compute the loss. """
        loss = self.common_step(batch, batch_nb)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        """ Need to overwrite validation_step to actually do validation. """
        loss = self.common_step(batch, batch_nb)
        return {'val_loss': loss}

    def validation_end(self, outputs):
        """ How to aggregate outputs of `validation_step` for logging."""
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        """ Required by pytorch-lightning. """
        return self.optimizer

    @pl.data_loader
    def train_dataloader(self):
        return self.loaders['train_loader']

    @pl.data_loader
    def val_dataloader(self):
        return self.loaders['val_loader']

    def on_save_checkpoint(self, checkpoint):
        return checkpoint
