from torch import nn, optim
from torch.utils import data
from pytorch_lightning import Trainer


from asteroid.engine.system import System
from asteroid.utils.test_utils import DummyDataset
from asteroid.engine.schedulers import NoamScheduler, DPTNetScheduler


def common_setup():
    model = nn.Sequential(nn.Linear(10, 10), nn.ReLU())
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    dataset = DummyDataset()
    loader = data.DataLoader(dataset, batch_size=2, num_workers=4)
    trainer = Trainer(max_epochs=1, fast_dev_run=True)
    return model, optimizer, loader, trainer


def test_state_dict():
    """Load and serialize scheduler."""
    model, optimizer, loader, trainer = common_setup()
    sched = NoamScheduler(optimizer, d_model=10, warmup_steps=100)
    state_dict = sched.state_dict()
    sched.load_state_dict(state_dict)
    state_dict_c = sched.state_dict()
    assert state_dict == state_dict_c
    # Test zero_grad
    sched.zero_grad()


def test_noam_scheduler():
    model, optimizer, loader, trainer = common_setup()
    scheduler = {
        "scheduler": NoamScheduler(optimizer, d_model=10, warmup_steps=100),
        "interval": "step",
    }

    system = System(
        model,
        optimizer,
        loss_func=nn.MSELoss(),
        train_loader=loader,
        val_loader=loader,
        scheduler=scheduler,
    )
    trainer.fit(system)
    # Test `as_tensor` for `plot`
    scheduler["scheduler"].as_tensor()


def test_dptnet_scheduler():
    model, optimizer, loader, trainer = common_setup()

    scheduler = {
        "scheduler": DPTNetScheduler(optimizer, d_model=10, steps_per_epoch=6, warmup_steps=4),
        "interval": "step",
    }

    system = System(
        model,
        optimizer,
        loss_func=nn.MSELoss(),
        train_loader=loader,
        val_loader=loader,
        scheduler=scheduler,
    )
    trainer.fit(system)
    # Test `as_tensor` for `plot`
    scheduler["scheduler"].as_tensor()
