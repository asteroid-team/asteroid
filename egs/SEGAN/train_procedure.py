from asteroid.engine.gan_system import GanSystem
import torch


class TrainSEGAN(GanSystem):
    """This class implements the training and validation procedure for SEGAN"""

    def training_step(self, batch, batch_nb, optimizer_idx):
        # Get data from data_loader
        inputs, targets = batch
        # Forward inputs
        estimates = self(inputs)
        # Train discriminator
        if optimizer_idx == 0:
            # Compute D loss for targets
            est_true_labels = self.discriminator(targets, inputs)
            true_loss = self.d_loss(est_true_labels, True)
            # Compute D loss for self.estimates
            est_false_labels = self.discriminator(estimates.detach(),
                                                  inputs)
            fake_loss = self.d_loss(est_false_labels, False)
            # Overall, the loss is the mean of these
            d_loss = (true_loss + fake_loss) * 0.5
            tqdm_dict = {'d_loss': d_loss}
            output = {
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            }
            return output

        # Train generator
        if optimizer_idx == 1:
            # The generator is supposed to fool the discriminator.
            est_labels = self.discriminator(estimates, inputs)
            adversarial_loss = self.g_loss(estimates, targets, est_labels)
            tqdm_dict = {'g_loss': adversarial_loss}
            output = {
                'loss': adversarial_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            }
            return output

    def validation_step(self, batch, batch_nb):
        """ Need to overwrite PL validation_step to do validation.

        Args:
            batch: the object returned by the loader (a list of torch.Tensor
                in most cases) but can be something else.
            batch_nb (int): The number of the batch in the epoch.

        Returns:
            dict:

            ``'val_loss'``: loss
        """
        inputs, targets = batch
        est_targets = self(inputs)
        val_loss = self.validation_loss(est_targets, targets)
        return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs):
        """ How to aggregate outputs of `validation_step` for logging.

        Args:
           outputs (list[dict]): List of validation losses, each with a
           ``'val_loss'`` key

        Returns:
            dict: Average loss

            ``'val_loss'``: Average loss on `outputs`

            ``'log'``: Tensorboard logs

            ``'progress_bar'``: Tensorboard logs
        """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs,
                'progress_bar': tensorboard_logs}
