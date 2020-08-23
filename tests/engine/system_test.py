from torch import nn, optim
from torch.utils import data
from pytorch_lightning import Trainer

from asteroid.engine.system import System
from asteroid.utils.test_utils import DummyDataset


def test_system():
    model = nn.Sequential(nn.Linear(10, 10), nn.ReLU())
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    dataset = DummyDataset()
    loader = data.DataLoader(dataset, batch_size=2, num_workers=4)
    system = System(
        model,
        optimizer,
        loss_func=nn.MSELoss(),
        train_loader=loader,
        val_loader=loader,
        scheduler=scheduler,
    )
    trainer = Trainer(max_epochs=1, fast_dev_run=True)
    trainer.fit(system)
