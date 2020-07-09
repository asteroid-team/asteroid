import argparse
import json
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from discriminator import make_discriminator_and_optimizer
from generator import make_generator_and_optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

# By default train.py will use all available GPUs. The `id` option in run.sh
# will limit the number of available GPUs for train.py .
# This can be changed: `python train.py --gpus 0,1` will only train on 2 GPUs.
from asteroid.data.metricGAN import MetricGAN
from asteroid.engine.gan_system import GanSystem
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from torch_stoi import NegSTOILoss

# Keys which are not in the conf_g.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]

parser = argparse.ArgumentParser()
parser.add_argument('--gpus', type=str, help='list of GPUs', default='-1')
parser.add_argument('--exp_dir', default='exp/tmp',
                    help='Full path to save best validation model')


def main(conf_g, conf_d):
    train_set = MetricGAN(csv_dir=conf_g['data']['train_dir'],
                          task=conf_g['data']['task'],
                          sample_rate=conf_g['data']['sample_rate'],
                          n_src=conf_g['data']['n_src'],
                          segment=conf_g['data']['segment'])

    val_set = MetricGAN(csv_dir=conf_g['data']['valid_dir'],
                        task=conf_g['data']['task'],
                        sample_rate=conf_g['data']['sample_rate'],
                        n_src=conf_g['data']['n_src'],
                        segment=conf_g['data']['segment'])

    train_loader = DataLoader(train_set, shuffle=True,
                              batch_size=conf_g['training']['batch_size'],
                              num_workers=conf_g['training']['num_workers'],
                              drop_last=True)

    val_loader = DataLoader(val_set, shuffle=False,
                            batch_size=conf_g['training']['batch_size'],
                            num_workers=conf_g['training']['num_workers'],
                            drop_last=True)

    generator, opt_g, g_loss = make_generator_and_optimizer(conf_g)
    discriminator, opt_d, d_loss = make_discriminator_and_optimizer(conf_d)

    # Define scheduler
    scheduler_g = None
    if conf_g['training']['half_lr']:
        scheduler_g = ReduceLROnPlateau(optimizer=opt_g, factor=0.5,
                                        patience=5)
    scheduler_d = None
    if conf_d['training']['half_lr']:
        scheduler_d = ReduceLROnPlateau(optimizer=opt_d, factor=0.5,
                                        patience=5)

    # Just after instantiating, save the args. Easy loading in the future.
    exp_dir = conf_g['main_args']['exp_dir']
    os.makedirs(exp_dir, exist_ok=True)
    conf_g_path = os.path.join(exp_dir, 'conf_g.yml')
    conf_d_path = os.path.join(exp_dir, 'conf_d.yml')
    with open(conf_g_path, 'w') as outfile:
        yaml.safe_dump(conf_g, outfile)
    with open(conf_d_path, 'w') as outfile:
        yaml.safe_dump(conf_d, outfile)

    validation_loss = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx')

    gan = GanSystem(discriminator=discriminator, generator=generator,
                    opt_d=opt_d, opt_g=opt_g, scheduler_d=scheduler_d,
                    scheduler_g=scheduler_g, discriminator_loss=d_loss,
                    generator_loss=g_loss, validation_loss=validation_loss,
                    train_loader=train_loader, val_loader=val_loader,
                    conf=conf_g)

    # Define callbacks
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints/')
    checkpoint = ModelCheckpoint(checkpoint_dir, monitor='val_loss',
                                 mode='min', save_top_k=5, verbose=True)

    early_stopping = False
    if conf_g['training']['early_stop']:
        early_stopping = EarlyStopping(monitor='val_loss', patience=10,
                                       verbose=True)

    # Don't ask GPU if they are not available.
    if not torch.cuda.is_available():
        print('No available GPU were found, set gpus to None')
        conf_g['main_args']['gpus'] = None

    trainer = pl.Trainer(max_epochs=conf_g['training']['epochs'],
                         checkpoint_callback=checkpoint,
                         early_stop_callback=early_stopping,
                         default_save_path=exp_dir,
                         gpus=conf_g['main_args']['gpus'],
                         distributed_backend='dp',
                         train_percent_check=1.0,
                         # Useful for fast experiment
                         gradient_clip_val=1.0)
    trainer.fit(gan)

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as file:
        json.dump(best_k, file, indent=0)


if __name__ == '__main__':
    import yaml
    from pprint import pprint
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    # We start with opening the conf_gig file conf_g.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    with open('local/conf_d.yml') as f:
        def_conf_d = yaml.safe_load(f)
    with open('local/conf_g.yml') as f:
        def_conf_g = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf_g, parser=parser)
    # Arguments are then parsed into a hierarchical dictionary (instead of
    # flat, as returned by argparse) to facilitate calls to the different
    # asteroid methods (see in main).
    # plain_args is the direct output of parser.parse_args() and contains all
    # the attributes in an non-hierarchical structure. It can be useful to also
    # have it so we included it here but it is not used.
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    pprint(arg_dic)
    pprint(def_conf_d)
    main(arg_dic, def_conf_d)
