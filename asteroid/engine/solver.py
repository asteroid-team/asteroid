"""
Class used to train models.
@author : Manuel Pariente, Inria-Nancy
Modified and extended from github.com/kaituoxu/Conv-TasNet/
MIT Copyright (c) 2018 Kaituo XU
"""

import os
import time
import torch

from ..utils import to_cuda

_defaults = {
    # Training defaults
    'epoch': 1,
    'half_lr': True,
    'early_stop': True,
    'max_norm': 5.,
    # Model defaults
    'checkpoint': False,
    'continue_from': None,
    # Logging
    'print_freq': 1000
}


class Solver(object):
    """ Training class.
    Args:
        loaders: {'train_loader': DataLoader, 'val_loader': DataLoader}.
        model: torch.nn.Module instance, the model to train.
        loss_class: class instance with a `compute` method.
            See asteroid.engine.losses for examples
        optimizer: torch.optim.Optimizer instance, used for training.
        model_path: String. Full path to save the models to.
        continue_from: String (optional), full path to model from which resume
            unfinished training.
        **training_conf: Keywork arguments such as in `defaults`. You can call
            `Solver.print_defaults()` to print the keys and default values.
    Methods:

    """
    def __init__(self, loaders, model, loss_class, optimizer, model_path,
                 continue_from=None, **training_conf):
        self.train_loader = loaders['train_loader']
        self.val_loader = loaders['val_loader']
        self.model = model
        self.loss_class = loss_class
        self.optimizer = optimizer
        self.model_path = model_path
        self.continue_from = continue_from
        self.tr_conf = dict(_defaults)  # Make a copy of the default values
        self.tr_conf.update(training_conf)

        self.save_folder = os.path.dirname(self.model_path)
        self.use_cuda = next(model.module.parameters()).is_cuda
        # Monitoring
        self.tr_loss = torch.zeros((self.tr_conf['epochs'], ))
        self.cv_loss = torch.zeros((self.tr_conf['epochs'], ))
        self.ep_half_lr = []
        self.best_val_loss = float("inf")
        self.val_no_impv = 0
        self.early_val_no_impv = 0
        self._reset()

    def _reset(self):
        """ Extract saved model from `continue_from` path. """
        if self.tr_conf['continue_from']:
            print('Loading checkpoint model '
                  '{}'.format(self.tr_conf['continue_from']))
            pack = torch.load(self.tr_conf['continue_from'])
            self.model.module.load_state_dict(pack['state_dict'])
            self.optimizer.load_state_dict(pack['optim_dict'])
            self.start_epoch = int(pack['infos'].get('epoch', 1))
            self.tr_loss[:self.start_epoch] = pack['tr_loss'][:self.start_epoch]
            self.cv_loss[:self.start_epoch] = pack['cv_loss'][:self.start_epoch]
        else:
            self.start_epoch = 0
        os.makedirs(self.save_folder, exist_ok=True)

    def train(self):
        for epoch in range(self.start_epoch, self.tr_conf['epochs']):
            print("Training...")
            self.model.train()
            start = time.time()
            tr_avg_loss = self.run_one_epoch(epoch)
            print('-' * 85)
            print('Train Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Train Loss {2:.3f}'.format(epoch + 1, time.time() - start,
                                              tr_avg_loss))
            print('-' * 85)
            if self.tr_conf['checkpoint']:
                file_path = os.path.join(self.save_folder,
                                         'epoch%d.pth.tar' % (epoch + 1))
                self.save_model(file_path, epoch)
            print('Cross validation...')
            self.model.eval()
            start = time.time()
            val_loss = self.run_one_epoch(epoch, validation=True)
            print('-' * 85)
            print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Valid Loss {2:.3f}'.format(epoch + 1, time.time() - start,
                                              val_loss))
            print('-' * 85)
            if val_loss > 1.001 * self.best_val_loss:  # (under 0.1%, no impv)
                self.val_no_impv += 1
                self.early_val_no_impv += 1
            else:
                self.val_no_impv = 0
                self.early_val_no_impv = 0
            # Adjust learning rate (halving)
            if self.tr_conf['half_lr'] and self.val_no_impv >= 3:
                self.multiply_learning_rate(self.optimizer, 0.5)
                self.ep_half_lr.append(epoch)
                self.val_no_impv = 0
            # Early stopping
            if self.early_val_no_impv >= 8 and self.tr_conf['early_stop']:
                print("No improvement for 8 epochs, early stopping.")
                break
            # Save the best model
            self.tr_loss[epoch] = tr_avg_loss
            self.cv_loss[epoch] = val_loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model(self.model_path, epoch)

    def run_one_epoch(self, epoch, validation=False):
        """ Run one epoch of training or validation."""
        start = time.time()
        total_loss = 0
        data_loader = self.train_loader if not validation else self.val_loader
        for i, (data) in enumerate(data_loader):
            loss = self.run_one_batch(data, validation=validation)
            total_loss += loss.item()
            if i % self.tr_conf['print_freq'] == 0:
                print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | Current'
                      ' Loss {3:.6f} | {4:.1f} ms/batch '
                      ''.format(epoch+1, i+1, total_loss/(i+1), loss.item(),
                                1000*(time.time()-start)/(i+1)), flush=True)
        return total_loss / (i + 1)

    def run_one_batch(self, data, validation=False):
        """ Run forward backward step for one batch."""
        inputs, targets, infos = self.unpack_data(data)
        est_targets = self.forward_model(inputs, validation=validation)
        loss = self.compute_loss(targets, est_targets, infos=infos)
        self.loss_backward_and_step(loss, validation=validation)
        return loss

    def unpack_data(self, data):
        """ Unpack data given by the DataLoader """
        if len(data) == 2:
            inputs, targets = data
            infos = dict()
        elif len(data) == 3:
            inputs, targets, infos = data
        else:
            raise ValueError('Expected DataLoader output to have '
                             '2 or 3 elements. Received '
                             '{} elements'.format(len(data)))
        if self.use_cuda:
            inputs, targets, infos = to_cuda([inputs, targets, infos])
        return inputs, targets, infos

    def forward_model(self, inputs, validation=False):
        """ Forwards the inputs through the model """
        if validation:
            with torch.no_grad():
                est_targets = self.model(inputs)
        else:
            est_targets = self.model(inputs)
        return est_targets

    def compute_loss(self, targets, est_targets, infos=None):
        """ Computes the loss to be backpropred.
        `infos` is a dictionary that is passed to the loss function.
        """
        return self.loss_class.compute(targets, est_targets, infos=infos)

    def loss_backward_and_step(self, loss, validation=False):
        """ Backpropagation of the loss and gradient descend step."""
        if not validation and not torch.isnan(loss):
            self.optimizer.zero_grad()
            loss.backward()
            # if isinstance(self.model.module.encoder, AnalyticFreeFB):
            #     value_if_nan(self.model.module.encoder.filters.grad)
            #     value_if_nan(self.model.module.decoder.filters.grad)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.tr_conf['max_norm'])
            self.optimizer.step()

    def save_model(self, file, epoch):
        torch.save(self.model.module.serialize(self.optimizer,
                                               tr_loss=self.tr_loss,
                                               cv_loss=self.cv_loss,
                                               half_lr=self.ep_half_lr,
                                               epoch=epoch), file)
        print('Saving model to {}'.format(file))

    @staticmethod
    def multiply_learning_rate(optimizer, multiplier):
        """Multiplies the learning rate of an optimizer by a given number.
        This is done inplace, so it does not return anything.

        Args:
            optimizer: torch.optim.Optimizer. The optimizer to be changed.
            multiplier: float > 0. Learning rate multiplier
        """
        for param_group in optimizer.param_groups:
            param_group['lr'] = multiplier * param_group['lr']
            print('Learning rate adjusted to: {}'.format(param_group['lr']))

    @staticmethod
    def print_defaults():
        print(_defaults)
