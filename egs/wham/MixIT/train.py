import os
import argparse
import json

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from asteroid import DPRNNTasNet
from asteroid.data.wham_dataset import WhamDataset
from asteroid.engine.optimizers import make_optimizer
from asteroid.engine.system import System
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid.losses import MixITLossWrapper, multisrc_neg_sisdr

# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]

# By default train.py will use all available GPUs. The `id` option in run.sh
# will limit the number of available GPUs for train.py .
parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", default="exp/tmp", help="Full path to save best validation model")


class MixITSystem(System):
    def training_step(self, batch, batch_nb):
        mixtures, oracle = batch

        bsz = mixtures.shape[0]
        mix1 = mixtures[bsz // 2 :]
        mix2 = mixtures[: bsz // 2]
        moms = mix1 + mix2

        est_sources = self(moms)
        loss = self.loss_func["mixit"](est_sources, torch.stack((mix1, mix2)).transpose(0, 1))

        with torch.no_grad():
            # we cheat during training and check oracle performance ^^
            loss_oracle = self.loss_func["pit"](
                est_sources, torch.cat((oracle[bsz // 2 :], oracle[: bsz // 2]), 1)
            )

        tensorboard_logs = {"train_loss": loss, "oracle_loss": loss_oracle}

        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_nb):

        mixture, oracle = batch
        est_sources = self(mixture)

        _, indxs = torch.sort(torch.sqrt(torch.mean(est_sources**2, dim=-1)), descending=True)
        indxs = indxs[:, :2]  # we know a-priori that in eval there are 2 sources,
        # we thus discard the estimates with lower energy.
        est_sources = est_sources.gather(1, indxs.unsqueeze(-1).repeat(1, 1, est_sources.shape[-1]))

        loss = self.loss_func["pit"](est_sources, oracle)
        return {"val_loss": loss}


def main(conf):

    assert (
        conf["training"]["batch_size"] % 2 == 0
    ), "Batch size must be divisible by two to run this recipe"

    train_set = WhamDataset(
        conf["data"]["train_dir"],
        "sep_clean",
        sample_rate=conf["data"]["sample_rate"],
        segment=conf["data"]["segment"],
        nondefault_nsrc=None,
    )
    val_set = WhamDataset(
        conf["data"]["valid_dir"],
        "sep_clean",
        sample_rate=conf["data"]["sample_rate"],
        nondefault_nsrc=None,
    )

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
    )

    model = DPRNNTasNet(
        **conf["filterbank"], **conf["masknet"], sample_rate=conf["data"]["sample_rate"]
    )
    optimizer = make_optimizer(model.parameters(), **conf["optim"])
    # Define scheduler
    scheduler = None
    if conf["training"]["half_lr"]:
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=5)
    # Just after instantiating, save the args. Easy loading in the future.
    exp_dir = conf["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    # Define Loss function.
    loss_func = {
        "pit": PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx"),
        "mixit": MixITLossWrapper(multisrc_neg_sisdr, generalized=True),
    }

    system = MixITSystem(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        config=conf,
    )

    # Define callbacks
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        checkpoint_dir, monitor="val_loss", mode="min", save_top_k=5, verbose=True
    )
    callbacks.append(checkpoint)
    if conf["training"]["early_stop"]:
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=30, verbose=True))

    trainer = pl.Trainer(
        max_epochs=conf["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy="ddp",
        devices="auto",
        gradient_clip_val=conf["training"]["gradient_clipping"],
    )
    trainer.fit(system)

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.model.serialize()
    to_save.update(train_set.get_infos())
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))


if __name__ == "__main__":
    import yaml
    from pprint import pprint
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
    pprint(arg_dic)
    main(arg_dic)
