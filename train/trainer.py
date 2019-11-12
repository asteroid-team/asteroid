import torch
import psutil
import collections
from catalyst.dl import utils
from config import ParamConfig
from memory_profiler import profile
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import EarlyStoppingCallback


@profile
def train(model: torch.nn.Module, dataset: torch.utils.data.Dataset,
          optimizer: torch.optim.Optimizer, criterion: torch.nn.Module,
          config: ParamConfig, validate: bool=False,
          val_dataset: torch.utils.data.Dataset=None, logdir: str="./logdir") -> None:
    """
        train the model with specified paremeters

        Args:
            model: neural network model
            dataset: training dataset
            optimizer: optimizer
            criterion: loss function
            validate: whether or not to validate
            val_dataset: validation dataset
    """
    
    loaders = collections.OrderedDict()
    train_loader = torch.utils.data.DataLoader(dataset, config.batch_size,
                                               shuffle=True, num_workers=config.workers)

    if val_dataset:
        val_loader = torch.utils.data.DataLoader(val_dataset, config.batch_size,
                                               shuffle=True, num_workers=config.workers)

    loaders["train"] = train_loader
    
    if validate:
        loaders["valid"] = val_loader

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5,
                                                           patience=2)

    for audio, video, mixed in train_loader:
        optimizer.zero_grad()
        output_audios = model(mixed, video)
        total_loss = criterion(output_audios, audio)
        total_loss.backward()
        optimizer.step()
        print(total_loss.item())
    runner = SupervisedRunner()
    runner.train(model=model, criterion=criterion,
                 optimizer=optimizer, scheduler=scheduler,
                 loaders=loaders, logdir=logdir, verbose=True,
                 num_epochs=config.epochs,
                 callbacks=[EarlyStoppingCallback(patience=2, min_delta=1e-2)]
                 )

    utils.plot_metrics(logdir=logdir, metrics=["loss", "_base/lr"])
