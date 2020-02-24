""" Train a deep clustering network
"""
import os
import argparse

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from asteroid.data.wsj0_mix import WSJ2mixDataset 
from asteroid.engine.system import System
from asteroid.losses import PITLossWrapper, pairwise_mse
from asteroid.losses import deep_clustering_loss

import asteroid.filterbanks as fb
from asteroid.filterbanks.inputs_and_masks import take_mag
from model import make_model_and_optimizer

EPS = torch.finfo(torch.float).eps

parser = argparse.ArgumentParser()
parser.add_argument('--gpus', type=str, help='list of GPUs', default='-1')
parser.add_argument('--exp_dir', default='exp/tmp',
                    help='Full path to save best validation model')

pit_loss = PITLossWrapper(pairwise_mse, mode='pairwise')

class DcSystem(System):
    """ System for deep clustering implementation. 
        Overwrites the common_step.
    """
    def __init__(self, model, optimizer, loss_func, train_loader,
                 val_loader=None, scheduler=None, config=None):
        super().__init__(model, optimizer, loss_func, train_loader,
                 val_loader=val_loader, scheduler=scheduler, config=config)
        self.enc = fb.Encoder(fb.STFTFB(**config['filterbank']))

    def common_step(self, batch, batch_nb):
        inputs, targets, masks = self.unpack_data(batch)
        est_targets = self(inputs)
        spec = take_mag(self.enc(inputs.unsqueeze(1)))
        loss = self.loss_func(est_targets, targets, masks, spec)
        return loss

    def unpack_data(self,batch):
        mix, sources = batch
        n_batch, n_src, n_sample = sources.shape
        new_sources = sources.view(-1, n_sample).unsqueeze(1)
        src_mag_spec = take_mag(self.enc(new_sources))
        fft_dim = src_mag_spec.shape[1]
        src_mag_spec = src_mag_spec.view(n_batch, n_src, fft_dim, -1)
        src_sum = src_mag_spec.sum(1).unsqueeze(1) + EPS
        real_mask = src_mag_spec/src_sum
        # Get the src idx having the maximum energy
        binary_mask = real_mask.argmax(1)
        return mix, binary_mask, real_mask


def handle_multiple_loss(est_heads, targets, true_real_masks, inputs, alpha=1):
    """ Handles deep clustering loss and the PIT loss
    Args:
        est_heads (tuple): Tuple containing embedding and estimated masks
        targets (np.array): Binary masks with one hot encoding
        true_real_masks (np.array): True real valued masks
        inputs(np.array): Spectrogram of the mixture
        alpha(int): Weight for the pit_loss

    Return:
        Sum of the deep clustering and PIT loss
    """
    embedding, est_masks = est_heads
    dc_loss = deep_clustering_loss(embedding, targets)
    pit_loss_batch = pit_loss(est_masks * inputs.unsqueeze(1),
                              true_real_masks * inputs.unsqueeze(1))
    return dc_loss.mean() + alpha * pit_loss_batch

def main(conf):
    train_set = WSJ2mixDataset(conf['data']['tr_wav_len_list'],
                               conf['data']['wav_base_path']+'/tr',
                               sample_rate=conf['data']['sample_rate'],
                               segment=conf['data']['segment'])
    val_set = WSJ2mixDataset(conf['data']['cv_wav_len_list'],
                             conf['data']['wav_base_path']+'/cv',
                             sample_rate=conf['data']['sample_rate'],
                             segment=conf['data']['segment'])
    train_loader = DataLoader(train_set, shuffle=True,
                              batch_size=conf['data']['batch_size'],
                              num_workers=conf['data']['num_workers'],
                              drop_last=True)
    val_loader = DataLoader(val_set, shuffle=True,
                            batch_size=conf['data']['batch_size'],
                            num_workers=conf['data']['num_workers'],
                            drop_last=True)
    model, optimizer = make_model_and_optimizer(conf)
    exp_dir = conf['main_args']['exp_dir']
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, 'conf.yml')
    with open(conf_path, 'w') as outfile:
        yaml.safe_dump(conf, outfile)
    loss_func = handle_multiple_loss
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints/')
    checkpoint = ModelCheckpoint(checkpoint_dir, monitor='val_loss',
                                 mode='min')
    system = DcSystem(model=model, loss_func=loss_func,
                      optimizer=optimizer,
                      train_loader=train_loader, val_loader=val_loader,
                      config=conf)
    # Don't ask GPU if they are not available.
    if not torch.cuda.is_available():
        print('No available GPU were found, set gpus to None')
        conf['main_args']['gpus'] = None
    trainer = pl.Trainer(max_nb_epochs=conf['training']['epochs'],
                         checkpoint_callback=checkpoint,
                         default_save_path=exp_dir,
                         gpus=conf['main_args']['gpus'],
                         distributed_backend='dp',
                         train_percent_check=1.0  # Useful for fast experiment
                        )
    trainer.fit(system)
    torch.save(system.model.state_dict(), os.path.join(exp_dir, 'final.pth'))

if __name__ == '__main__':
    import yaml
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    with open('conf.yml') as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    print(arg_dic)
    main(arg_dic)
