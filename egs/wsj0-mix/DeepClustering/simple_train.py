import os
import argparse
import json
from ipdb import set_trace

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim

from asteroid.filterbanks.transforms import take_mag
import asteroid.filterbanks as fb
from asteroid.data.wsj0_mix import WSJ2mixDataset, BucketingSampler, \
        collate_fn
from asteroid.masknn.blocks import SingleRNN
from asteroid.losses import PITLossWrapper, pairwise_mse
from asteroid.losses import deep_clustering_loss

EPS = torch.finfo(torch.float32).eps
enc = fb.Encoder(fb.STFTFB(256, 256, stride=64))
enc = enc.cuda()

parser = argparse.ArgumentParser()
parser.add_argument('--gpus', type=str, help='list of GPUs', default='-1')
parser.add_argument('--exp_dir', default='exp/tmp',
                    help='Full path to save best validation model')

pit_loss = PITLossWrapper(pairwise_mse, mode='pairwise')

class Model(nn.Module):
    def __init__(self):
    #def __init__(self, in_chan, n_src, rnn_type = 'lstm',
    #        embedding_dim=20, n_layers=2, hidden_size=600,
    #        dropout=0, bidirectional=True, log=False):
        super(Model, self).__init__()
        in_chan = 129
        rnn_type = 'lstm'
        self.input_dim = in_chan
        n_layers = 2
        hidden_size = 600
        bidirectional = True
        dropout = 0.5
        embedding_dim = 20
    #    self.n_src = n_src
        self.embedding_dim = embedding_dim
        self.rnn = SingleRNN(rnn_type, in_chan, hidden_size, n_layers, \
            dropout, bidirectional)
        self.dropout = nn.Dropout(dropout)
        rnn_out_dim = hidden_size * 2 if bidirectional else hidden_size
    #    self.embedding_layer = nn.Linear(rnn_out_dim, \
    #            in_chan * embedding_dim)
        self.embedding_layer = nn.Linear(rnn_out_dim, \
                in_chan * embedding_dim)
        self.mask_layer = nn.Linear(rnn_out_dim, in_chan * 2)
        self.non_linearity = nn.Sigmoid()
        self.EPS = torch.finfo(torch.float32).eps
    #    if log:
    #        #TODO: Use pytorch lightning logger here
    #       print('Using log spectrum as input')
        # Temp check
        self.lin1 = nn.Linear(in_chan, in_chan)
        self.lin2 = nn.Linear(in_chan, in_chan)

    def forward(self, input_data):
        batches, freq_dim, seq_cnt = input_data.shape
        out = self.rnn(input_data.permute(0,2,1))
        out = self.dropout(out)
        mask_out = self.mask_layer(out)
        mask_out = mask_out.view(batches, seq_cnt, 2,
                self.input_dim).permute(0, 2, 3, 1) 
        mask_out = self.non_linearity(mask_out)

        projection = self.embedding_layer(out)
        projection = self.non_linearity(projection)
        projection = projection.view(batches, -1, self.embedding_dim)
        proj_norm = torch.norm(projection, p=2, dim=-1, keepdim=True) + \
                torch.finfo(torch.float32).eps
        projection_final =  projection/proj_norm
        #return None, mask_out[...,:self.input_dim]
        return projection_final, mask_out

class Model1(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        in_chan = 129
        self.lin1 = nn.Linear(in_chan, in_chan)
        self.lin2 = nn.Linear(in_chan, in_chan)
        self.non_linearity = nn.Sigmoid()

    def forward(self, input_data):
        out = self.lin1(input_data.permute(0,2,1))
        out = self.non_linearity(out)
        out = self.lin2(out)
        out = self.non_linearity(out)
        return  None, out

def unpack_data(batch):
    mix, sources = batch
    n_batch, n_src, n_sample = sources.shape
    new_sources = sources.view(-1, n_sample).unsqueeze(1)
    src_mag_spec = take_mag(enc(new_sources))
    fft_dim = src_mag_spec.shape[1]
    src_mag_spec = src_mag_spec.view(n_batch, n_src, fft_dim, -1)
    src_sum = src_mag_spec.sum(1).unsqueeze(1) + EPS
    real_mask = src_mag_spec/src_sum
    # Get the src idx having the maximum energy
    binary_mask = real_mask.argmax(1)
    return mix, binary_mask, real_mask

def compute_cost(model, batch):
    inputs, targets, masks = unpack_data(batch)
    spec = take_mag(enc(inputs.unsqueeze(1)))
    #spec = take_mag(enc(batch[1][:,0,:].unsqueeze(1)))
    spec = spec.cuda()
    est_targets = model(spec)
    #masks = torch.stack((masks[:,0,...], masks[:,0,...]),dim=1)
    #masks = masks[:,0,...].permute(0,2,1)
    #masks = masks.permute(0,2,1)
    masks = masks.cuda()
    #temp = torch.rand(5,129, 300)
    #temp = temp.cuda()
    #est_targets = model(temp)
    #masks = temp.permute(0,2,1)
    #torch.save((masks.data.cpu(), spec.data.cpu()), 'mask_spec.pt')
    #loss = torch.sqrt(torch.pow(est_targets[1] - masks, 2)+EPS).mean()
    #loss = torch.pow(est_targets[1] - masks, 2).mean()
    #loss = pairwise_mse(est_targets[1], masks).mean()
    loss = pit_loss(est_targets[1], masks)
    embedding = est_targets[0]

    vad_tf_mask =  compute_vad(spec).cuda()
    targets = targets.cuda()
    dc_loss = deep_clustering_loss(embedding, targets, 
            binary_mask=vad_tf_mask)
    #loss = torch.sqrt(torch.pow(est_targets[1] - spec.permute(0,2,1), 2)).mean()
    return dc_loss, loss

def compute_vad(spectra, threshold_db=40):
    ''' Compute a time-frequency VAD
    source: github.com/funcwj/deep-clustering.git
    '''
    # to dB
    spectra_db = 20 * torch.log10(spectra)
    max_magnitude_db = torch.max(spectra_db)
    threshold = 10**((max_magnitude_db - threshold_db) / 20)
    mask = spectra > threshold
    return mask.double()

def main(conf):
    train_set = WSJ2mixDataset(conf['data']['tr_wav_len_list'],
                               conf['data']['wav_base_path']+'/tr',
                               sample_rate=conf['data']['sample_rate'])
    val_set = WSJ2mixDataset(conf['data']['cv_wav_len_list'],
                             conf['data']['wav_base_path']+'/cv',
                             sample_rate=conf['data']['sample_rate'])
    train_set.shuffle_list()
    val_set.shuffle_list()
    #train_set.id_list = train_set.id_list[:5]
    #val_set.id_list = val_set.id_list[:5]
    #train_set.len = 5
    #val_set.len = 5

    train_sampler = BucketingSampler(train_set,
                                     batch_size=conf['data']['batch_size'])
    valid_sampler = BucketingSampler(val_set,
                                     batch_size=conf['data']['batch_size'])

    train_loader = DataLoader(train_set, 
                              batch_sampler=train_sampler,
                              collate_fn=collate_fn,
                              num_workers=conf['data']['num_workers'])
    val_loader = DataLoader(val_set, 
                            batch_sampler=valid_sampler,
                            collate_fn=collate_fn,
                            num_workers=conf['data']['num_workers'])

    model = Model().cuda()
    optimizer = optim.Adam(model.parameters())

    for _ in range(50):
        train_loss = 0
        dc_loss_all = 0
        pit_loss_all = 0
        for batch_nb, temp_batch in enumerate(train_loader):
            batch_nb += 1
            batch = [el.cuda() for el in temp_batch]
            #batch = temp_batch
            model.train()
            optimizer.zero_grad()
            dc_loss, pit_loss_val = compute_cost(model, batch)
            if np.isnan(dc_loss.item()):
                print('\n')
                set_trace()
                dc_loss, pit_loss_val = compute_cost(model, batch)
            dc_loss *= 1e-4
            loss = dc_loss + pit_loss_val

            train_loss += loss.item()
            dc_loss_all += dc_loss.item()
            pit_loss_all += pit_loss_val.item()

            #loss.backward()
            pit_loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            print('{}\t total:{:06.4f}\t dc:{:06.4f}\t pit:{:06.4f}'.format(batch_nb, train_loss/(batch_nb), dc_loss_all/(batch_nb), pit_loss_all/batch_nb),
                    end='\r')
        print('Train loss {}\t total:{:06.4f}\t dc:{:06.4f}\t pit:{:06.4f}'.format(batch_nb, train_loss/(batch_nb), dc_loss_all/(batch_nb), pit_loss_all/batch_nb))

        train_loss = 0
        dc_loss_all = 0
        pit_loss_all = 0
        for batch_nb, temp_batch in enumerate(val_loader):
        #for batch_nb, temp_batch in enumerate(train_loader):
            batch = [el.cuda() for el in temp_batch]
            #batch = temp_batch
            model.eval()
            dc_loss, pit_loss_val = compute_cost(model, batch)
            loss = dc_loss + pit_loss_val

            train_loss += loss.item()
            dc_loss_all += dc_loss.item()
            pit_loss_all += pit_loss_val.item()
            print('{}\t total:{:06.4f}\t dc:{:06.4f}\t pit:{:06.4f}'.format(batch_nb, train_loss/(batch_nb), dc_loss_all/(batch_nb), pit_loss_all/batch_nb))
        print('Valid loss {}\t total:{:06.4f}\t dc:{:06.4f}\t pit:{:06.4f}'.format(batch_nb, train_loss/(batch_nb), dc_loss_all/(batch_nb), pit_loss_all/batch_nb))

if __name__ == '__main__':
    import yaml
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    with open('local/conf.yml') as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    set_trace()
    print(arg_dic)
    main(arg_dic)
