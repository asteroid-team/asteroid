import os
import argparse
from torch.utils.data import DataLoader
from asteroid.data.wham_dataset import WhamDataset
from asteroid.engine.system import System
from model import make_model_and_optimizer

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

parser = argparse.ArgumentParser()
parser.add_argument('--gpus', type=str, help='list of GPUs', default='-1')
parser.add_argument('--exp_dir', default='exp/tmp',
                    help='Full path to save best validation model')


def main(conf):
    # from asteroid.data.toy_data import WavSet
    # train_set = WavSet(n_ex=1000, n_src=2, ex_len=32000)
    # val_set = WavSet(n_ex=1000, n_src=2, ex_len=32000)
    # Define data pipeline
    train_set = WhamDataset(conf['data']['train_dir'], conf['data']['task'],
                            sample_rate=conf['data']['sample_rate'],
                            nondefault_nsrc=conf['data']['nondefault_nsrc'])
    val_set = WhamDataset(conf['data']['valid_dir'], conf['data']['task'],
                          sample_rate=conf['data']['sample_rate'],
                          nondefault_nsrc=conf['data']['nondefault_nsrc'])

    train_loader = DataLoader(train_set, shuffle=True,
                              batch_size=conf['data']['batch_size'],
                              num_workers=conf['data']['num_workers'])
    val_loader = DataLoader(val_set, shuffle=True,
                            batch_size=conf['data']['batch_size'],
                            num_workers=conf['data']['num_workers'])
    conf['masknet'].update({'n_src': train_set.n_src})

    # Define model and optimizer in a local function (defined in the recipe).
    # Two advantages to this : re-instantiating the model and optimizer
    # for retraining and evaluating is straight-forward.
    model, optimizer = make_model_and_optimizer(conf)

    # Just after instantiating, save the args. Easy loading in the future.
    exp_dir = conf['main_args']['exp_dir']
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, 'conf.yml')
    with open(conf_path, 'w') as outfile:
        yaml.safe_dump(conf, outfile)

    # Define Loss function.
    loss_class = PITLossContainer(pairwise_neg_sisdr, n_src=train_set.n_src)
    # Checkpointing callback can monitor any quantity which is returned by
    # validation step, defaults to val_loss here (see System).
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints/')
    checkpoint = ModelCheckpoint(checkpoint_dir, monitor='val_loss',
                                 mode='min', save_best_only=False)
    # New PL version will come the 7th of december / will have save_top_k
    system = System(model=model, loss_class=loss_class, optimizer=optimizer,
                    train_loader=train_loader, val_loader=val_loader,
                    config=conf)
    trainer = pl.Trainer(max_nb_epochs=conf['training']['epochs'],
                         checkpoint_callback=checkpoint,
                         default_save_path=exp_dir,
                         gpus=conf['main_args']['gpus'],
                         distributed_backend='dp')
    trainer.fit(system)


if __name__ == '__main__':
    import yaml
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    with open('conf.yml') as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)

    # Arg_dic is a dictionary following the structure of `conf.yml`
    # plain_args is the direct output of parser.parse_args() and contains all
    # the attributes in an non-hierarchical structure.
    main(arg_dic)
