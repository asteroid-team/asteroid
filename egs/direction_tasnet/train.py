import os
import argparse
import json

import torch
import pytorch_lightning as pl
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from asteroid.models import save_publishable
from local.tac_dataset import TACDataset
from torch.nn import MSELoss
from asteroid.engine.optimizers import make_optimizer
from asteroid.engine.system import System
from conv_tasnet import TasNet
from dataset import make_dataloader
from torch.nn import L1Loss
from preprocess import Prep, n_sp

# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]

# By default train.py will use all available GPUs. The `id` option in run.sh
# will limit the number of available GPUs for train.py.
parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", default="exp/tmp", help="Full path to save best validation model")


class AngleSystem(System):
    def common_step(self, batch, batch_nb, train=True):
        data = batch
        targets = torch.stack(data["ref"]).transpose(1, 0).squeeze()
        #valid_channels = torch.ones(len(inputs), 1) * 6
        #valid_channels = valid_channels.to(dtype=torch.long, device=inputs.device)
        # valid_channels contains a list of valid microphone channels for each example.
        # each example can have a varying number of microphone channels (can come from different arrays).
        # e.g. [[2], [4], [1]] three examples with 2 mics 4 mics and 1 mics.
        ests_list = []
        inputs = data["mix"]
        for i in range(n_sp):
             est_targets = self.model(inputs, data[i])
             ests_list.append(est_targets)
        est = torch.cat(ests_list, dim=1)
        loss = self.loss_func(est, targets)  # first channel is used as ref
        return loss


def main(conf):
    '''
    train_set = TACDataset(conf["data"]["train_json"], conf["data"]["segment"], train=True)
    val_set = TACDataset(conf["data"]["dev_json"], conf["data"]["segment"], train=False)

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
    '''

    train_loader = make_dataloader(train=True,
                                   batch_size=conf['training']["batch_size"],
                                   chunk_size=conf['data']['chunk'],
                                   num_workers=conf['training']['num_workers'])
    val_loader = make_dataloader(train=False,
                                 batch_size=conf['training']['batch_size'],
                                 chunk_size=conf['data']['chunk'],
                                 num_workers=conf['training']['num_workers'])
    #Prep(train_loader)
    #Prep(val_loader)
    #for data in train_loader:
        #print(type(data[0]))


    model = TasNet()
    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # params = sum([np.prod(p.size()) for p in model_parameters])
    # print(params)
    # exit()
    optimizer = make_optimizer(model.parameters(), **conf["optim"])
    # Define scheduler
    if conf["training"]["half_lr"]:
        scheduler = ReduceLROnPlateau(
            optimizer=optimizer, factor=0.5, patience=conf["training"]["patience"]
        )
    else:
        scheduler = None
    # Just after instantiating, save the args. Easy loading in the future.
    exp_dir = conf["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    # Define Loss function.
    loss_func = MSELoss()
    system = AngleSystem(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        config=conf,
    )

    # Define callbacks
    # Define callbacks
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor="val_loss",
        mode="min",
        save_top_k=conf["training"]["save_top_k"],
        verbose=True,
    )
    callbacks.append(checkpoint)
    if conf["training"]["early_stop"]:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss", mode="min", patience=conf["training"]["patience"], verbose=True
            )
        )

    # Don't ask GPU if they are not available.
    gpus = [-1]
    trainer = pl.Trainer(
        max_epochs=conf["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        #gpus=gpus,
        distributed_backend="ddp",
        gradient_clip_val=conf["training"]["gradient_clipping"],
    )
    trainer.fit(system)

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    #to_save = system.model.serialize()
    #to_save.update(train_set.get_infos())
    torch.save(system.model.state_dict(), os.path.join(exp_dir, "best_model.ckpt"))
    #save_publishable(
        #os.path.join(exp_dir, "publish_dir"),
        #to_save,
        #metrics=dict(),
        #train_conf=conf,
        #recipe="asteroid/TAC",
    #)


if __name__ == "__main__":
    import yaml
    from pprint import pprint as print
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict



    # We start with opening the config file conf.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    with open("./local/conf.yml") as f:
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
