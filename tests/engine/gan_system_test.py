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
    def __init__(self):
        super().__init__()

    def forward(self, estimates, targets, est_labels):
        loss = torch.mean((est_labels - torch.ones_like(est_labels)) ** 2)
        return loss


class DiscriminatorLoss(_Loss):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, estimates, est_labels, labels):
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
        self.model = nn.Sequential(nn.Linear(10, 1),
                                   nn.Sigmoid())

    def forward(self, x, y, z):
        """
        Forward pass of discriminator.
        Args:
            x: batch of estimates
            y: batch of targets
            z: batch of inputs
        """
        return self.model(x)


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
    gan = GanSystem(discriminator=discriminator, generator=generator,
                    opt_d=opt_d, opt_g=opt_g, scheduler_d=scheduler_d,
                    scheduler_g=scheduler_g, discriminator_loss=d_loss,
                    generator_loss=g_loss, validation_loss=validation_loss,
                    train_loader=loader, val_loader=loader)
    trainer = Trainer(max_epochs=1, fast_dev_run=True)
    trainer.fit(gan)
