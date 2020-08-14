import os
import argparse
import json

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from asteroid.models import ConvTasNet
from asteroid.data import LibriMix
from asteroid.engine.optimizers import make_optimizer
from asteroid.engine.system import System
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr

# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]

# By default train.py will use all available GPUs. The `id` option in run.sh
# will limit the number of available GPUs for train.py .
parser = argparse.ArgumentParser()
parser.add_argument('--exp_dir', default='exp/tmp',
                    help='Full path to save best validation model')


def main(conf):
    train_set = LibriMix(csv_dir=conf['data']['train_dir'],
                         task=conf['data']['task'],
                         sample_rate=conf['data']['sample_rate'],
                         n_src=conf['data']['n_src'],
                         segment=conf['data']['segment'])

    val_set = LibriMix(csv_dir=conf['data']['valid_dir'],
                       task=conf['data']['task'],
                       sample_rate=conf['data']['sample_rate'],
                       n_src=conf['data']['n_src'],
                       segment=conf['data']['segment'])

    train_loader = DataLoader(train_set, shuffle=True,
                              batch_size=conf['training']['batch_size'],
                              num_workers=conf['training']['num_workers'],
                              drop_last=True)

    val_loader = DataLoader(val_set, shuffle=False,
                            batch_size=conf['training']['batch_size'],
                            num_workers=conf['training']['num_workers'],
                            drop_last=True)
    conf['masknet'].update({'n_src': conf['data']['n_src']})

    model = ConvTasNet(**conf['filterbank'], **conf['masknet'])
    optimizer = make_optimizer(model.parameters(), **conf['optim'])
    # Define scheduler
    scheduler = None
    if conf['training']['half_lr']:
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5,
                                      patience=5)
    # Just after instantiating, save the args. Easy loading in the future.
    exp_dir = conf['main_args']['exp_dir']
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, 'conf.yml')
    with open(conf_path, 'w') as outfile:
        yaml.safe_dump(conf, outfile)

    # Define Loss function.
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx')
    system = System(model=model, loss_func=loss_func, optimizer=optimizer,
                    train_loader=train_loader, val_loader=val_loader,
                    scheduler=scheduler, config=conf)

    # Define callbacks
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints/')
    checkpoint = ModelCheckpoint(checkpoint_dir, monitor='val_loss',
                                 mode='min', save_top_k=5, verbose=1)
    early_stopping = False
    if conf['training']['early_stop']:
        early_stopping = EarlyStopping(monitor='val_loss', patience=30,
                                       verbose=1)

    # Don't ask GPU if they are not available.
    gpus = -1 if torch.cuda.is_available() else None
    trainer = pl.Trainer(max_epochs=conf['training']['epochs'],
                         checkpoint_callback=checkpoint,
                         early_stop_callback=early_stopping,
                         default_save_path=exp_dir,
                         gpus=gpus,
                         distributed_backend='dp',
                         train_percent_check=1.0,  # Useful for fast experiment
                         gradient_clip_val=5.)
    trainer.fit(system)

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    # Save best model (next PL version will make this easier)
    best_path = [b for b, v in best_k.items() if v == min(best_k.values())][0]
    state_dict = torch.load(best_path)
    system.load_state_dict(state_dict=state_dict['state_dict'])
    system.cpu()

    to_save = system.model.serialize()
    to_save.update(train_set.get_infos())
    torch.save(to_save, os.path.join(exp_dir, 'best_model.pth'))


if __name__ == '__main__':
    import yaml
    from pprint import pprint as print
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    # We start with opening the config file conf.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    with open('local/conf.yml') as f:
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
