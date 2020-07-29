import torch
from torch import nn, optim
from torch.utils import data
from pytorch_lightning import Trainer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from asteroid.engine.gan_system import GanSystem
from torch.nn.modules.loss import _Loss
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr


class DummyDataset(data.Dataset):
    def __init__(self):
        self.inp_dim = 10
        self.out_dim = 10

    def __len__(self):
        return 20

    def __getitem__(self, idx):
        return torch.randn(1, self.inp_dim), torch.randn(1, self.out_dim)


class GeneratorLoss(_Loss):
    def forward(self, est_labels):
        loss = torch.mean((est_labels - torch.ones_like(est_labels)) ** 2)
        return loss


class DiscriminatorLoss(_Loss):
    def forward(self, est_labels, labels):
        # Behaves differently if estimates come from  the generated data or not
        if labels:
            loss = torch.mean((est_labels - torch.ones_like(est_labels)) ** 2)
        else:
            loss = torch.mean(est_labels ** 2)
        return loss


class Discriminator(nn.Module):
    """D"""
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(10, 1), nn.Sigmoid())

    def forward(self, x):
        """
        Forward pass of discriminator.
        Args:
            x: batch of estimates
        """
        return self.model(x)


class TrainGAN (GanSystem):

    def training_step(self, batch, batch_nb, optimizer_idx):
        # Get data from data_loader
        inputs, targets = batch
        # Forward inputs
        estimates = self(inputs)
        # Train discriminator
        if optimizer_idx == 0:
            # Compute D loss for targets
            est_true_labels = self.discriminator(targets)
            true_loss = self.d_loss(est_true_labels, True)
            # Compute D loss for self.estimates
            est_false_labels = self.discriminator(estimates.detach())
            fake_loss = self.d_loss(est_false_labels, False)
            # Overall, the loss is the mean of these
            d_loss = (true_loss + fake_loss) * 0.5
            tqdm_dict = {'d_loss': d_loss}
            output = {
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            }
            return output

        # Train generator
        if optimizer_idx == 1:
            # The generator is supposed to fool the discriminator.
            est_labels = self.discriminator(estimates)
            adversarial_loss = self.g_loss(est_labels)
            tqdm_dict = {'g_loss': adversarial_loss}
            output = {
                'loss': adversarial_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            }
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

    def validation_epoch_end(self, outputs):
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


def test_system():
    discriminator = Discriminator()
    generator = nn.Sequential(nn.Linear(10, 10), nn.ReLU())
    opt_d = optim.Adam(discriminator.parameters(), lr=1e-3)
    opt_g = optim.Adam(generator.parameters(), lr=1e-3)
    scheduler_d = ReduceLROnPlateau(optimizer=opt_d, factor=0.5, patience=5)
    scheduler_g = ReduceLROnPlateau(optimizer=opt_g, factor=0.5,  patience=5)
    g_loss = GeneratorLoss()
    d_loss = DiscriminatorLoss()
    validation_loss = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx')
    dataset = DummyDataset()
    loader = data.DataLoader(dataset, batch_size=4, num_workers=4)
    gan = TrainGAN(discriminator=discriminator, generator=generator,
                    opt_d=opt_d, opt_g=opt_g,  discriminator_loss=d_loss,
                    generator_loss=g_loss, validation_loss=validation_loss,
                    train_loader=loader, val_loader=loader,
                    scheduler_d=scheduler_d, scheduler_g=scheduler_g)
    trainer = Trainer(max_epochs=1, fast_dev_run=True)
    trainer.fit(gan)
