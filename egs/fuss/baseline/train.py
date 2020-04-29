import os
import argparse
import json

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from asteroid.engine.system import System
from asteroid.data import FUSSDataset
from asteroid.losses import PITLossWrapper, PairwiseNegSDR

from model import make_model_and_optimizer


# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]

# By default train.py will use all available GPUs. The `id` option in run.sh
# will limit the number of available GPUs for train.py .
# This can be changed: `python train.py --gpus 0,1` will only train on 2 GPUs.
parser = argparse.ArgumentParser()
parser.add_argument('--exp_dir', default='exp/tmp',
                    help='Full path to save best validation model')


EPS = 1e-8
class ClipedSingleSrcSNR(nn.Module):
    def __init__(self, active_clip=-30, inactive_clip=-20):
        super().__init__()
        self.active_clip = active_clip
        self.inactive_clip = inactive_clip

    def forward(self, est, target, mix=None):
        loss = target.new_zeros(target.shape[0],)
        target_norm = torch.norm(target, p=2, dim=-1)
        silent_mask = target_norm < EPS  # (B,)
        active_loss = - self.active_snr(est, target)[~silent_mask]
        loss[~silent_mask] = torch.clamp(active_loss, min=self.active_clip)
        # loss /= (1 - silent_mask.float().mean()) + EPS
        inactive_loss = - self.inactive_snr(est, mix)[silent_mask]
        loss[silent_mask] = torch.clamp(inactive_loss, min=self.inactive_clip)
        return loss

    @staticmethod
    def active_snr(est, target):
        den = torch.norm(target - est, dim=-1, p=2) + EPS
        return 20. * torch.log10(EPS + torch.norm(target, dim=-1, p=2) / den)

    @staticmethod
    def inactive_snr(est, mix):
        den = torch.norm(est, dim=-1, p=2) + EPS
        return 20. * torch.log10(EPS + torch.norm(mix, dim=-1, p=2) / den)


class ClipedNegSNR(PairwiseNegSDR):
    """ Clipped SNR loss as described the sound-separation repo's README. """
    def __init__(self, active_clip=-30, inactive_clip=-20):
        super().__init__(sdr_type='snr', zero_mean=True, take_log=True)
        self.active_clip = active_clip
        self.inactive_clip = inactive_clip

    def forward(self, est_targets, targets, mix=None):
        # Get pairwise losses for SNR
        pwl = super(ClipedNegSNR, self).forward(est_targets, targets)
        # Compute silent targets and make an index mask out of it
        # Don't forget pwl is (batch, estimates, targets)
        silent = (torch.norm(targets, dim=-1, p=2) < 1e-8).unsqueeze(1)
        # If all sources are active, don't both the select and scatter
        if (~silent).all():
            return pwl.clamp(min=self.active_clip)

        silent_pwl = silent.expand_as(pwl)
        # Clip non-silent SNR to active_clip.
        pwl.masked_scatter_(~silent_pwl, torch.clamp(pwl[~silent_pwl],
                                                     min=self.active_clip))
        # Recompute silent SNR (it is -10 *log(EPS)=80)
        # We use the sum of all targets to compute a reference power.
        # We want the zero output to have negligible power with respect to the
        # total references. Clip to inactie_clip
        e_noise = torch.norm(est_targets, dim=-1).unsqueeze(-1)
        if mix is None:
            ref_power = targets.sum(1, keepdim=True).norm(dim=-1, p=2,
                                                          keepdim=True)
        else:
            ref_power = mix.norm(dim=-1, p=2, keepdim=True)
        power_ratio = e_noise.expand_as(silent_pwl)[silent_pwl] / (ref_power+EPS)
        to_scatter = 20 * torch.log10(power_ratio + EPS)
        pwl.masked_scatter_(silent_pwl, torch.clamp(to_scatter,
                                                    min=self.inactive_clip))
        return pwl


class FUSSSystem(System):
    def common_step(self, batch, batch_nb, train=True):
        inputs, targets, bg = batch
        est_targets = self(inputs, bg=bg)
        # Some mixture only have background. We need it to compute the loss
        loss = self.loss_func(est_targets, targets, mix=inputs)
        return loss


def main(conf):
    # Define loaders
    train_set = FUSSDataset(conf['data']['train_list'], return_bg=True)
    val_set = FUSSDataset(conf['data']['valid_list'], return_bg=True)

    train_loader = DataLoader(train_set, shuffle=True,
                              batch_size=conf['training']['batch_size'],
                              num_workers=conf['training']['num_workers'],
                              drop_last=True)
    val_loader = DataLoader(val_set, shuffle=True,
                            batch_size=conf['training']['batch_size'],
                            num_workers=conf['training']['num_workers'],
                            drop_last=True)

    # Define model and optimizer in a local function (defined in the recipe).
    # Two advantages to this : re-instantiating the model and optimizer
    # for retraining and evaluating is straight-forward.
    model, optimizer = make_model_and_optimizer(conf)
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
    # loss_func = PITLossWrapper(ClipedNegSNR(), pit_from='pw_mtx')
    loss_func = PITLossWrapper(ClipedSingleSrcSNR(), pit_from='pw_pt')

    system = FUSSSystem(model=model, loss_func=loss_func, optimizer=optimizer,
                        train_loader=train_loader, val_loader=val_loader,
                        scheduler=scheduler, config=conf)

    # Define callbacks
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints/')
    checkpoint = ModelCheckpoint(checkpoint_dir, monitor='val_loss',
                                 mode='min', save_top_k=5, verbose=1)
    early_stopping = False
    if conf['training']['early_stop']:
        early_stopping = EarlyStopping(monitor='val_loss', patience=10,
                                       verbose=1)

    gpus=-1  # By default, take all GPUs (set through --id in recipe)
    # Don't ask GPU if they are not available.
    if not torch.cuda.is_available():
        print('No available GPU were found, set gpus to None')
        gpus=None
    trainer = pl.Trainer(max_epochs=conf['training']['epochs'],
                         checkpoint_callback=checkpoint,
                         early_stop_callback=early_stopping,
                         default_save_path=exp_dir,
                         gpus=gpus,
                         distributed_backend='dp',
                         train_percent_check=1,  # Useful for fast experiment
                         gradient_clip_val=5.)
    trainer.fit(system)

    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(checkpoint.best_k_models, f, indent=0)


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
