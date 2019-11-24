import os

import torch
import argparse
import yaml

from torch.utils.data import DataLoader
from torch import nn

from asteroid import Container, Solver
import asteroid.filterbanks as fb
from asteroid.masknn import TDConvNet
from asteroid.engine.losses import PITLossContainer, pairwise_neg_sisdr
from asteroid.data.wham_dataset import WhamDataset
from asteroid.engine.optimizers import make_optimizer

from model import make_model_and_optimizer

# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]
parser = argparse.ArgumentParser()
parser.add_argument('--use_cuda', type=int, default=0,
                    help='Whether use GPU')
parser.add_argument('--model_path', default='exp/tmp/final.pth',
                    help='Full path to save best validation model')


def main(conf):
    # Define data pipeline with datasets and loaders
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
    loaders = {'train_loader': train_loader, 'val_loader': val_loader}
    # Update the number of sources given by the task, if not overwritten.
    conf['masknet'].update({'nsrc': train_set.n_src})
    # Define model and optimizerin a local function.
    model, optimizer = make_model_and_optimizer(conf)

    # Just after instantiating, save the args
    conf_path = os.path.join(os.path.dirname(conf['main_args']['model_path']),
                             'conf.yml')
    with open(conf_path) as outfile:
        yaml.safe_dump(outfile, conf, defaul_flow_style=False)

    if conf['main_args']['use_cuda']:
        model.cuda()
    # Define Loss function
    loss_class = PITLossContainer(pairwise_neg_sisdr, n_src=train_set.n_src)
    # Pass everything to the solver with a training dicitonary defined in
    # the conf.yml file. Finally, call .train() and that's it.
    solver = Solver(loaders, model, loss_class, optimizer,
                    model_path=conf['main_args']['model_path'],
                    **conf['training'])
    solver.train()


if __name__ == '__main__':
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    with open('conf.yml') as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)

    main(arg_dic)
