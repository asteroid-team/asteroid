import torch
import argparse
from torch.utils.data import DataLoader
from torch import nn

from asteroid import Container, Solver
from asteroid.filterbanks import FreeFB, ParamSincFB, STFTFB, AnalyticFreeFB
from asteroid.masknn import TDConvNet
from asteroid.engine.losses import PITLossContainer, pairwise_neg_sisdr
from asteroid.data.wham_dataset import WhamDataset

parser = argparse.ArgumentParser()
parser.add_argument('--use_cuda', type=int, default=0,
                    help='Whether use GPU')
# parser.add_argument('--num_workers', default=4, type=int,
#                     help='Number of workers to generate minibatch')
# save and load model
parser.add_argument('--save_folder', default='exp/temp',
                    help='Location to save epoch models')
parser.add_argument('--checkpoint', dest='checkpoint', default=0, type=int,
                    help='Enables checkpoint saving of model')
parser.add_argument('--continue_from', default='',
                    help='Continue from checkpoint model')
parser.add_argument('--model_path', default='final.pth.tar',
                    help='Location to save best validation model')
# logging
parser.add_argument('--print_freq', default=1000, type=int,
                    help='Frequency of printing training infomation')


def main(args):
    # Save training_conf dictionary

    # Define data pipeline
    train_set = WhamDataset(args['data']['train_dir'], args['data']['task'],
                            sample_rate=args['data']['sample_rate'],
                            nondefault_nsrc=args['data']['nondefault_nsrc'])
    val_set = WhamDataset(args['data']['valid_dir'], args['data']['task'],
                          sample_rate=args['data']['sample_rate'],
                          nondefault_nsrc=args['data']['nondefault_nsrc'])

    train_loader = DataLoader(train_set, shuffle=True,
                              batch_size=args['data']['batch_size'],
                              num_workers=args['data']['num_workers'])
    val_loader = DataLoader(val_set, shuffle=True,
                              batch_size=args['data']['batch_size'],
                              num_workers=args['data']['num_workers'])
    data = {'train_loader': train_loader, 'val_loader': val_loader}

    # Define model
    encoder = FreeFB(enc_or_dec='encoder', **args['filterbank'])
    decoder = FreeFB(enc_or_dec='decoder', **args['filterbank'])
    masker = TDConvNet(in_chan=encoder.n_feats_out, out_chan=encoder.n_feats_out,
                       n_src=train_set.n_src, **args['masknet'])
    model = nn.DataParallel(Container(encoder, masker, decoder))
    # Transfer to cuda

    # Define Loss functions
    loss_class = PITLossContainer(pairwise_neg_sisdr, n_src=train_set.n_src)
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args['optim']['lr'],
                                 weight_decay=args['optim']['weight_decay'])

    # Pass everything to the solver (expects a dictionary for now)
    solver = Solver(data, model, loss_class, optimizer, args['training'])
    solver.train()


if __name__ == '__main__':
    import yaml
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    with open('conf.yml') as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    arg_dic = parse_args_as_dict(parser)

    # temp modification, waiting for run.sh
    arg_dic['training'].update(arg_dic['optional arguments'])
    main(arg_dic)
