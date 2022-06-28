import os
import argparse
import json

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from asteroid.data.wham_dataset import WhamDataset
from system import SystemTwoStep
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr, PairwiseNegSDR

from model import get_encoded_paths
from model import load_best_filterbank_if_available
from model import make_model_and_optimizer

# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]

# By default train.py will use all available GPUs. The `id` option in run.sh
# will limit the number of available GPUs for train.py .
parser = argparse.ArgumentParser()
parser.add_argument(
    "--exp_dir", default="exp/model_logs", help="Full path to save best validation model"
)


def get_data_loaders(conf, train_part="filterbank"):
    train_set = WhamDataset(
        conf["data"]["train_dir"],
        conf["data"]["task"],
        sample_rate=conf["data"]["sample_rate"],
        nondefault_nsrc=conf["data"]["nondefault_nsrc"],
        normalize_audio=True,
    )
    val_set = WhamDataset(
        conf["data"]["valid_dir"],
        conf["data"]["task"],
        sample_rate=conf["data"]["sample_rate"],
        nondefault_nsrc=conf["data"]["nondefault_nsrc"],
        normalize_audio=True,
    )

    if train_part not in ["filterbank", "separator"]:
        raise ValueError("Part to train: {} is not available.".format(train_part))

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        drop_last=True,
        batch_size=conf[train_part + "_training"][train_part[0] + "_batch_size"],
        num_workers=conf[train_part + "_training"][train_part[0] + "_num_workers"],
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        drop_last=True,
        batch_size=conf[train_part + "_training"][train_part[0] + "_batch_size"],
        num_workers=conf[train_part + "_training"][train_part[0] + "_num_workers"],
    )
    # Update number of source values (It depends on the task)
    conf["masknet"].update({"n_src": train_set.n_src})

    return train_loader, val_loader


def train_model_part(conf, train_part="filterbank", pretrained_filterbank=None):
    train_loader, val_loader = get_data_loaders(conf, train_part=train_part)

    # Define model and optimizer in a local function (defined in the recipe).
    # Two advantages to this : re-instantiating the model and optimizer
    # for retraining and evaluating is straight-forward.
    model, optimizer = make_model_and_optimizer(
        conf, model_part=train_part, pretrained_filterbank=pretrained_filterbank
    )
    # Define scheduler
    scheduler = None
    if conf[train_part + "_training"][train_part[0] + "_half_lr"]:
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=5)
    # Just after instantiating, save the args. Easy loading in the future.
    exp_dir, checkpoint_dir = get_encoded_paths(conf, train_part)
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    # Define Loss function.
    loss_func = PITLossWrapper(PairwiseNegSDR("sisdr", zero_mean=False), pit_from="pw_mtx")
    system = SystemTwoStep(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        config=conf,
        module=train_part,
    )

    # Define callbacks
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        checkpoint_dir, monitor="val_loss", mode="min", save_top_k=1, verbose=True
    )
    callbacks.append(checkpoint)
    if conf[train_part + "_training"][train_part[0] + "_early_stop"]:
        callbacks.append(EarlyStopping(monitor="val_loss", patience=30, verbose=True))

    trainer = pl.Trainer(
        max_epochs=conf[train_part + "_training"][train_part[0] + "_epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy="ddp",
        devices="auto",
        limit_train_batches=1.0,  # Useful for fast experiment
        gradient_clip_val=5.0,
    )
    trainer.fit(system)

    with open(os.path.join(checkpoint_dir, "best_k_models.json"), "w") as file:
        json.dump(checkpoint.best_k_models, file, indent=0)


def main(conf):
    filterbank = load_best_filterbank_if_available(conf)
    _, checkpoint_dir = get_encoded_paths(conf, "filterbank")
    if filterbank is None:
        print(
            "There are no available filterbanks under: {}. Going to "
            "training.".format(checkpoint_dir)
        )
        train_model_part(conf, train_part="filterbank")
        filterbank = load_best_filterbank_if_available(conf)
    else:
        print("Found available filterbank at: {}".format(checkpoint_dir))
        if not conf["filterbank_training"]["reuse_pretrained_filterbank"]:
            print("Refining filterbank...")
            train_model_part(conf, train_part="filterbank")
            filterbank = load_best_filterbank_if_available(conf)
    train_model_part(conf, train_part="separator", pretrained_filterbank=filterbank)


if __name__ == "__main__":
    import yaml
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    # We start with opening the config file conf.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    with open("local/conf.yml") as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    # Arguments are then parsed into a hierarchical dictionary (instead of
    # flat, as returned by argparse) to facilitate calls to the different
    # asteroid methods (see in main).
    # plain_args is the direct output of parser.parse_args() and contains all
    # the attributes in an non-hierarchical structure. It can be useful to also
    # have it so we included it here but it is not used.
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    print(arg_dic)
    main(arg_dic)
