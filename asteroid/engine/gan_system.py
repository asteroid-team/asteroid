import pytorch_lightning as pl
import torch


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
                 validation_loss=None, val_loader=None,
                 scheduler_d=None, scheduler_g=None,
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

    def forward(self, *args, **kwargs):
        return self.generator(*args, **kwargs)

    def training_step(self, batch, batch_nb, optimizer_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_nb):
        """ Overwrite if needed. Called by pytorch-lightning"""
        pass

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
