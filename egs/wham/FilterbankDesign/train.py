import torch
import argparse
from torch.utils.data import DataLoader
from torch import nn

from asteroid import Container, Solver
from asteroid import filterbanks
from asteroid.masknn import TDConvNet
from asteroid.engine.losses import PITLossContainer, pairwise_neg_sisdr
from asteroid.data.wham_dataset import WhamDataset
from asteroid.engine.optimizers import make_optimizer

parser = argparse.ArgumentParser()
parser.add_argument('--filterbank_type', default='free')
parser.add_argument('--use_cuda', type=int, default=0,
                    help='Whether use GPU')
parser.add_argument('--model_path', default='exp/tmp/final.pth',
                    help='Full path to save best validation model')


def main(conf):
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
    loaders = {'train_loader': train_loader, 'val_loader': val_loader}

    # Define model
    # The encoder and decoder can directly be made from the dictionary.
    encoder, decoder = filterbanks.make_enc_dec(**conf['filterbank'])

    # The input post-processing changes the dimensions of input features to
    # the mask network. Different type of masks impose different output
    # dimensions to the mask network's output. We correct for these here.
    nn_in = int(encoder.n_feats_out * encoder.in_chan_mul)
    nn_out = int(encoder.n_feats_out * encoder.out_chan_mul)
    masker = TDConvNet(in_chan=nn_in, out_chan=nn_out,
                       n_src=train_set.n_src, **conf['masknet'])
    # The model is defined in Container, which is passed to DataParallel.
    model = nn.DataParallel(Container(encoder, masker, decoder))
    if conf['main_args']['use_cuda']:
        model.cuda()

    # Define Loss function : Here we use time domain SI-SDR.
    loss_class = PITLossContainer(pairwise_neg_sisdr, n_src=train_set.n_src)
    # Define optimizer : can be instantiate from dictonary as well.
    optimizer = make_optimizer(model.parameters(), **conf['optim'])

    # Pass everything to the solver and train
    solver = Solver(loaders, model, loss_class, optimizer,
                    model_path=conf['main_args']['model_path'],
                    **conf['training'])
    # solver.train()
    solver.run_one_epoch(0, validation=True)


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
