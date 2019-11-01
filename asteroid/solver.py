"""
Class used to train models.
@author : Manuel Pariente, Inria-Nancy
Modified and extended from github.com/kaituoxu/Conv-TasNet/
"""

import os
import time
import torch

from .utils import to_cuda


class Solver(object):
    """
    Args:
        data:
        model:
        loss_class:
        optimizer:
        args:
    """
    def __init__(self, data, model, loss_class, optimizer, args):

        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        self.model = model
        self.loss_class = loss_class
        self.optimizer = optimizer

        # Training config
        self.use_cuda = args.use_cuda
        self.epochs = args.epochs
        self.half_lr = args.half_lr
        self.early_stop = args.early_stop
        self.max_norm = args.max_norm
        # Save and load model
        self.save_folder = args.save_folder
        self.checkpoint = args.checkpoint
        self.continue_from = args.continue_from
        self.model_path = args.model_path
        # Monitoring
        self.print_freq = args.print_freq
        self.tr_loss = torch.as_tensor(self.epochs)
        self.cv_loss = torch.as_tensor(self.epochs)
        self.ep_half_lr = []
        self.best_val_loss = float("inf")
        self.val_no_impv = 0
        self.early_val_no_impv = 0
        self._reset()

    def _reset(self):
        """ Extract saved model from `continue_from` path. """
        if self.continue_from:
            print('Loading checkpoint model {}'.format(self.continue_from))
            pack = torch.load(self.continue_from)
            self.model.module.load_state_dict(pack['state_dict'])
            self.optimizer.load_state_dict(pack['optim_dict'])
            self.start_epoch = int(pack.get('epoch', 1))
            self.tr_loss[:self.start_epoch] = pack['tr_loss'][:self.start_epoch]
            self.cv_loss[:self.start_epoch] = pack['cv_loss'][:self.start_epoch]
        else:
            self.start_epoch = 0
        os.makedirs(self.save_folder, exist_ok=True)

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            print("Training...")
            self.model.train()
            start = time.time()
            tr_avg_loss = self._run_one_epoch(epoch)
            print('-' * 85)
            print('Train Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Train Loss {2:.3f}'.format(epoch + 1, time.time() - start,
                                              tr_avg_loss))
            print('-' * 85)
            if self.checkpoint:
                file_path = os.path.join(self.save_folder,
                                         'epoch%d.pth.tar' % (epoch + 1))
                self.save_model(file_path, epoch)
            print('Cross validation...')
            self.model.eval()
            start = time.time()
            val_loss = self._run_one_epoch(epoch, cross_valid=True)
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
            if self.half_lr and self.val_no_impv >= 3:
                self.multiply_learning_rate(self.optimizer, 0.5)
                self.ep_half_lr.append(epoch)
                self.val_no_impv = 0
            # Early stopping
            if self.early_val_no_impv >= 8 and self.early_stop:
                print("No improvement for 8 epochs, early stopping.")
                break
            # Save the best model
            self.tr_loss[epoch] = tr_avg_loss
            self.cv_loss[epoch] = val_loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                file_path = os.path.join(self.save_folder, self.model_path)
                self.save_model(file_path, epoch)

    def _run_one_epoch(self, epoch, cross_valid=False):
        start = time.time()
        total_loss = 0
        data_loader = self.tr_loader if not cross_valid else self.cv_loader
        for i, (data) in enumerate(data_loader):
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
            if cross_valid:
                with torch.no_grad():
                    est_targets = self.model(inputs)
            else:
                est_targets = self.model(inputs)
            loss = self.loss_class.compute(targets, est_targets, infos=infos)

            if not cross_valid and not torch.isnan(loss):
                self.optimizer.zero_grad()
                loss.backward()
                # if isinstance(self.model.module.encoder, AnalyticFreeFB):
                #     value_if_nan(self.model.module.encoder.filters.grad)
                #     value_if_nan(self.model.module.decoder.filters.grad)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.max_norm)
                self.optimizer.step()

            total_loss += loss.item()

            if i % self.print_freq == 0:
                print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | Current'
                      ' Loss {3:.6f} | {4:.1f} ms/batch '
                      ''.format(epoch+1, i+1, total_loss/(i+1), loss.item(),
                                1000*(time.time()-start)/(i+1)), flush=True)
        return total_loss / (i + 1)

    def save_model(self, file, epoch):
        torch.save(self.model.module.serialize(self.model.module,
                                               self.optimizer,
                                               epoch + 1,
                                               tr_loss=self.tr_loss,
                                               cv_loss=self.cv_loss,
                                               half_lr=self.ep_half_lr), file)
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
