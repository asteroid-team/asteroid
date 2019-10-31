import torch
import collections
from catalyst.dl import utils
from config import ParamConfig
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import EarlyStoppingCallback


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

    """
    for _ in range(1):#audio, video, mixed in train_loader:
        mixed = torch.zeros((1, 2, 298, 257))
        video = [torch.zeros((1, 1024, 75, 1)), torch.zeros((1, 1024, 75, 1))]
        audio = [torch.zeros((1, 298, 257, 2)), torch.zeros((1, 298, 257, 2))]

        output_audios = model(mixed, video)
        print(output_audios.shape)
        total_loss = 0
        for i in range(dataset.input_audio_size):
            loss = criterion(output_audios[:, i, ...], audio[i])
            total_loss += 0
        print(total_loss)
    """
    runner = SupervisedRunner()
    runner.train(model=model, criterion=criterion,
                 optimizer=optimizer, scheduler=scheduler,
                 loaders=loaders, logdir=logdir, verbose=True,
                 num_epochs=config.epochs,
                 callbacks=[EarlyStoppingCallback(patience=2, min_delta=1e-2)]
                 )

    utils.plot_metrics(logdir=logdir, metrics=["loss", "_base/lr"])
