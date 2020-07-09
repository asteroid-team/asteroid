from collections import OrderedDict
import soundfile as sf
import pytorch_lightning as pl
import torch
import os
from torch_stoi import NegSTOILoss
from asteroid.losses import PITLossWrapper


class GanSystem(pl.LightningModule):

    """ Base class for GAN training systems.
       Contains a generator a discriminator, two optimizers,
       a generator loss function, a discriminator loss function,
       a validation loss function,
       training and validation dataloaders and learning rate scheduler.

       Args:
           discriminator (torch.nn.Module): Instance of discriminator (d).
           generator (torch.nn.Module): Instance of generator (g).
           opt_d (torch.optim.Optimizer): Instance of optimizer for d.
           opt_g (torch.optim.Optimizer): Instance of optimizer for g.
           discriminator_loss (callable): Loss function with signature
               (est_targets, targets).
            generator_loss (callable): Loss function with signature
               (est_targets, targets).
           train_loader (torch.utils.data.DataLoader): Training dataloader.
           val_loader (torch.utils.data.DataLoader): Validation dataloader.
           scheduler_d (torch.optim.lr_scheduler._LRScheduler): Instance
               of learning rate schedulers for d.
           scheduler_g (torch.optim.lr_scheduler._LRScheduler): Instance
               of learning rate schedulers for g.
           conf: Anything to be saved with the checkpoints during training.
               The config dictionary to re-instantiate the run for example.
       """

    def __init__(self, discriminator, generator, opt_d, opt_g,
                 discriminator_loss, generator_loss, train_loader,
                 validation_loss=PITLossWrapper(NegSTOILoss(), pit_from='pw_pt'),
                 val_loader=None, scheduler_d=None, scheduler_g=None,
                 conf=None):

        super(GanSystem, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.opt_d = opt_d
        self.opt_g = opt_g
        self.d_loss = discriminator_loss
        self.g_loss = generator_loss
        self.validation_loss = validation_loss
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler_d = scheduler_d
        self.scheduler_g = scheduler_g
        self.conf = conf
        self.estimates = None

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_nb, optimizer_idx):
        # Get data from data_loader
        inputs, targets = batch
        # Forward inputs
        self.estimates = self(inputs)
        # Train discriminator
        if optimizer_idx == 0:
            # Compute D loss for targets
            est_true_labels = self.discriminator(targets, inputs, targets)
            true_loss = self.d_loss(inputs, targets, self.estimates,
                                    est_true_labels, True)
            # Compute D loss for self.estimates
            est_false_labels = self.discriminator(self.estimates.detach(),
                                                  inputs, targets)
            fake_loss = self.d_loss(inputs, targets, self.estimates,
                                    est_false_labels, False)
            # Overall, the loss is the mean of these
            d_loss = (true_loss + fake_loss) * 0.5
            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # Train generator
        if optimizer_idx == 1:
            # The generator is supposed to fool the discriminator.
            est_labels = self.discriminator(self.estimates, inputs, targets)
            adversarial_loss = self.g_loss(self.estimates, targets, est_labels)
            tqdm_dict = {'g_loss': adversarial_loss}
            output = OrderedDict({
                'loss': adversarial_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

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
        inputs, targets = batch
        est_targets = self(inputs)
        val_loss = self.validation_loss(est_targets, targets)
        return {'val_loss': val_loss}

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

    def configure_optimizers(self):
        opt_g = self.opt_g
        opt_d = self.opt_d
        scheduler_d = self.scheduler_d
        scheduler_g = self.scheduler_g

        if scheduler_d is not None and scheduler_g is not None:
            return [opt_d, opt_g], [scheduler_d, scheduler_g]

        elif scheduler_d is None and scheduler_g is not None:
            return [opt_d, opt_g], [scheduler_g]

        elif scheduler_d is not None and scheduler_g is None:
            return [opt_d, opt_g], [scheduler_d]

        return [opt_d, opt_g]

    def on_batch_start(self, batch):
        """ Overwrite if needed. Called by pytorch-lightning"""
        pass

    def on_batch_end(self):
        """ Overwrite if needed. Called by pytorch-lightning"""
        pass

    def on_epoch_start(self):
        """ Overwrite if needed. Called by pytorch-lightning"""
        pass

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
