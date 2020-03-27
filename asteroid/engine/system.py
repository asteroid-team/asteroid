import torch
import pytorch_lightning as pl
from argparse import Namespace
from ..utils import flatten_dict


class System(pl.LightningModule):
    """ Base class for deep learning systems.
    Contains a model, an optimizer, a loss function, training and validation
    dataloaders and learning rate scheduler.

    Args:
        model (torch.nn.Module): Instance of model.
        optimizer (torch.optim.Optimizer): Instance or list of optimizers.
        loss_func (callable): Loss function with signature
            (est_targets, targets).
        train_loader (torch.utils.data.DataLoader): Training dataloader.
        val_loader (torch.utils.data.DataLoader): Validation dataloader.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Instance, or list
            of learning rate schedulers.
        config: Anything to be saved with the checkpoints during training.
            The config dictionary to re-instantiate the run for example.
        two_step_approach (str):
            None: then the two step approach is not used.
            'separator': The two step is used for trainign the separation
                         module only.
            'filterbank': The two step approach is used for training only the
                          adaptive encoder/decoder or in other words the
                          filterbank.
            For more info take a look at method common_step_two_step_separtion()
    .. note:: By default, `training_step` (used by `pytorch-lightning` in the
        training loop) and `validation_step` (used for the validation loop)
        share `common_step`. If you want different behavior for the training
        loop and the validation loop, overwrite both `training_step` and
        `validation_step` instead.
    """
    def __init__(self, model, optimizer, loss_func, train_loader,
                 val_loader=None, scheduler=None, config=None,
                 two_step_approach=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        if two_step_approach is not None:
            assert (two_step_approach == 'filterbank' or
                    two_step_approach == 'separator'), 'If the two-step ' \
                   'approach is used then either filterbank or separator has ' \
                   'to be used but got: {}'.format(two_step_approach)
        self.two_step_approach = two_step_approach
        config = {} if config is None else config
        self.config = config
        # hparams will be logged to Tensorboard as text variables.
        # torch doesn't support None in the summary writer for now, convert
        # None to strings temporarily.
        # See https://github.com/pytorch/pytorch/issues/33140
        self.hparams = Namespace(**self.none_to_string(flatten_dict(config)))

    def forward(self, *args, **kwargs):
        """ Applies forward pass of the model.

        Returns:
            :class:`torch.Tensor`
        """
        return self.model(*args, **kwargs)

    def common_step(self, batch, batch_nb, train=True):
        """ Common forward step between training and validation.

        The function of this method is to unpack the data given by the loader,
        forward the batch through the model and compute the loss.
        Pytorch-lightning handles all the rest.

        Args:
            batch: the object returned by the loader (a list of torch.Tensor
                in most cases) but can be something else.
            batch_nb (int): The number of the batch in the epoch.
            train (bool): Whether in training mode. Needed only if the training
                and validation steps are fundamentally different, otherwise,
                pytorch-lightning handles the usual differences.

        Returns:
            :class:`torch.Tensor` : The loss value on this batch.

        .. note:: This is typically the method to overwrite when subclassing
            `System`. If the training and validation steps are somehow
            different (except for loss.backward() and optimzer.step()),
            the argument `train` can be used to switch behavior.
            Otherwise, `training_step` and `validation_step` can be overwriten.
        """
        inputs, targets = batch
        est_targets = self(inputs)
        loss = self.loss_func(est_targets, targets)
        return loss

    def common_step_two_step_separtion(self, batch, mode='train'):
        """ Common forward step between training and validation.

        The function of this method is to unpack the data given by the loader,
        forward the batch through the model and compute the loss for the
        separation module and the filterbank when the optimization process
        for source separation is breaken in two distinct processes as
        proposed in [1].

        Args:
            batch: the object returned by the loader (a list of torch.Tensor
                in most cases) but can be something else.
            mode (str): 'train' or 'val'. In case of training or validation the
                filterbank will return the estimated time signals. However,
                in training mode the separator will be trained using the
                ideal latent targets and it will estimate the corresponding
                latent representations of the sources as proposed in [1].
        Returns:
            dict:

            ``'loss'``: loss

            ``'log'``: dict with tensorboard logs

        References:
            [1]: Tzinis, E., Venkataramani, S., Wang, Z., Subakan, Y. C., and
                 Smaragdis, P., "Two-Step Sound Source Separation:
                 Training on Learned Latent Targets." In Acoustics, Speech
                 and Signal Processing (ICASSP), 2020 IEEE International
                 Conference. https://arxiv.org/abs/1910.09804
        """
        mixture_time, true_sources_time = batch
        if self.two_step_approach == 'filterbank':
            est_sources_time, _ = self(mixture_time, true_sources_time)
            return self.loss_func(est_sources_time, true_sources_time)

        # Here we train or validate the separator. In training we need the
        # latent targets to regress on. In validation we just provide the
        # estimated time domain signals.
        if mode == 'train':
            latent_targets = self.model.get_ideal_latent_targets(
                mixture_time, true_sources_time)
            est_latents = self.model.estimate_latent_representations(
                mixture_time)
            batch_size, n_sources = est_latents.shape[0], est_latents.shape[1]
            return self.loss_func(
                est_latents.view(batch_size, n_sources, -1),
                latent_targets.view(batch_size, n_sources, -1))
        elif mode == 'val':
            est_sources_time = self.model(mixture_time)
            return self.loss_func(est_sources_time, true_sources_time)
        else:
            raise NotImplementedError('The requested mode: {} is not '
                                      'available. Expected `train` or `val`.')

    def training_step(self, batch, batch_nb):
        """ Pass data through the model and compute the loss.

        Backprop is **not** performed (meaning PL will do it for you).

        Args:
            batch: the object returned by the loader (a list of torch.Tensor
                in most cases) but can be something else.
            batch_nb (int): The number of the batch in the epoch.

        Returns:
            dict:

            ``'loss'``: loss

            ``'log'``: dict with tensorboard logs

        """
<<<<<<< 1645d5ea3c9512ea841eb83cf88295ae9bc073a7
        loss = self.common_step(batch, batch_nb, train=True)
=======
        if self.two_step_approach is not None:
            loss = self.common_step_two_step_separtion(batch, mode='train')
        else:
            loss = self.common_step(batch, batch_nb)
>>>>>>> Pytorch Lightning System module overwrite commonstep method for integrating the two step source separation recipe. Specifically, for the two step process a new method callback is called which has a modular structure and does not intervene with the other callback methods existing.
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        """ Need to overwrite PL validation_step to do validation.

        Args:
            batch: the object returned by the loader (a list of torch.Tensor
                in most cases) but can be something else.
            batch_nb (int): The number of the batch in the epoch.

        Returns:
            dict:

            ``'val_loss'``: loss
        """
        loss = self.common_step(batch, batch_nb, train=False)
        return {'val_loss': loss}

    def validation_end(self, outputs):
        """ How to aggregate outputs of `validation_step` for logging.

        Args:
           outputs (list[dict]): List of validation losses, each with a
           ``'val_loss'`` key

        Returns:
            dict: Average loss

            ``'val_loss'``: Average loss on `outputs`

            ``'log'``: Tensorboard logs

            ``'progress_bar'``: Tensorboard logs
        """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs,
                'progress_bar': tensorboard_logs}

    def unsqueeze_if_dp_or_ddp(self, *values):
        """ Apply unsqueeze(0) to all values if training is done with dp
            or ddp. Unused now."""
        if self.trainer.use_dp or self.trainer.use_ddp2:
            values = [v.unsqueeze(0) for v in values]
        if len(values) == 1:
            return values[0]
        return values

    def configure_optimizers(self):
        """ Required by pytorch-lightning. """
        if self.scheduler is not None:
            return [self.optimizer], [self.scheduler]
        return self.optimizer

    @pl.data_loader
    def train_dataloader(self):
        return self.train_loader

    @pl.data_loader
    def val_dataloader(self):
        return self.val_loader

    @pl.data_loader
    def tng_dataloader(self):  # pragma: no cover
        """ Deprecated."""
        pass

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

    @staticmethod
    def none_to_string(dic):
        """ Converts `None` to  ``'None'`` to be handled by torch summary writer.

        Args:
            dic (dict): Dictionary to be transformed.

        Returns:
            dict: Transformed dictionary.
        """
        for k, v in dic.items():
            if v is None:
                dic[k] = str(v)
        return dic
