from collections import OrderedDict
import soundfile as sf
import pytorch_lightning as pl
import torch
import os


class GanSystem(pl.LightningModule):

    def __init__(self, routine, discriminator, generator, opt_d, opt_g,
                 scheduler_d, scheduler_g, discriminator_loss, validation_loss,
                 train_loader, special_g_loss=None, val_loader=None):

        super(GanSystem, self).__init__()
        self.routine = routine
        self.discriminator = discriminator
        self.generator = generator
        self.opt_d = opt_d
        self.opt_g = opt_g
        self.discriminator_loss = discriminator_loss
        self.validation_loss = validation_loss
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler_d = scheduler_d
        self.scheduler_g = scheduler_g
        if special_g_loss is None:
            self.adversarial_loss = discriminator_loss
        else:
            self.adversarial_loss = special_g_loss

    def forward(self, z):
        return self.generator(z)

    def routine_step(self, batch, optimizer_idx, routine):

        if routine == 'SEGAN':

            # Get data from data_loader
            inputs, targets = batch
            inputs = inputs.unsqueeze(1)
            # Forward inputs
            generated_sounds = self(inputs)
            # The discriminator is basically a binary classifier
            true_labels = True
            false_labels = False

            # Train discriminator
            if optimizer_idx == 0:
                # Compute D loss for targets (labels = 1)
                true_loss = self.discriminator_loss(self.discriminator(
                    torch.cat((targets, inputs), dim=1)), true_labels)
                # Compute D loss for generated sounds (labels = 0)
                fake_loss = self.discriminator_loss(
                    self.discriminator(torch.cat((generated_sounds.detach(),
                                       inputs), dim=1)), false_labels)
                # Overall, the loss is the mean of these
                d_loss = (true_loss + fake_loss) * 0.5
                tqdm_dict = {'d_loss': d_loss}
                output = OrderedDict({
                    'loss': d_loss,
                    'progress_bar': tqdm_dict,
                    'log': tqdm_dict
                })
                return output

            # Train generator
            if optimizer_idx == 1:
                # The generator is supposed to fool the discriminator.
                # To do so labels are set as true.
                g_loss = self.adversarial_loss(generated_sounds,
                                               targets,
                                               self.discriminator(torch.cat((
                                                   generated_sounds, inputs),
                                                   dim=1)))
                tqdm_dict = {'g_loss': g_loss}
                output = OrderedDict({
                    'loss': g_loss,
                    'progress_bar': tqdm_dict,
                    'log': tqdm_dict
                })
                return output

    def training_step(self, batch, batch_nb, optimizer_idx):
        # Reference for Virtual Batch Normalization
        output = self.routine_step(batch, optimizer_idx, self.routine)
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
        inputs = inputs.unsqueeze(1)
        est_targets = self(inputs)
        val_loss = self.validation_loss(est_targets, targets)
        # return {'val_loss': val_loss}
        return {'val_loss': val_loss}

    def validation_end(self, outputs):
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

    def configure_optimizers(self):
        opt_g = self.opt_g
        opt_d = self.opt_d
        scheduler_d = self.scheduler_d
        scheduler_g = self.scheduler_g

        if scheduler_d is not None and scheduler_g is not None:
            return [opt_d, opt_g], [scheduler_d, scheduler_g]

        elif scheduler_d is None and scheduler_g is not None:
            return [opt_d, opt_g], [scheduler_g]

        elif scheduler_d is not None and scheduler_g is None:
            return [opt_d, opt_g], [scheduler_d]

        return [opt_d, opt_g]

    # def on_save_checkpoint(self, checkpoint):
    #     """ Overwrite if you want to save more things in the checkpoint."""
    #     checkpoint['training_config'] = self.conf_g
    #     return checkpoint

    def on_batch_start(self, batch):
        """ Overwrite if needed. Called by pytorch-lightning"""
        pass

    def on_batch_end(self):
        """ Overwrite if needed. Called by pytorch-lightning"""
        pass

    def on_epoch_start(self):
        """ Overwrite if needed. Called by pytorch-lightning"""
        pass

    def on_epoch_end(self):
        """ Overwrite if needed. Called by pytorch-lightning"""
        sample_rate = 16000
        inputs, targets = next(iter(self.val_loader))
        inputs = inputs.unsqueeze(1)
        generated_outputs = self(inputs.cuda())
        noisy = inputs[1].cpu().squeeze().numpy()
        clean = targets[1].cpu().squeeze().numpy()
        generated = generated_outputs[1].detach().cpu().squeeze().numpy()
        path = os.path.join("res/", f"{self.current_epoch}")
        os.makedirs(path)
        sf.write(os.path.join(path, "clean.wav"), clean, sample_rate)
        sf.write(os.path.join(path, "noisy.wav"), noisy, sample_rate)
        sf.write(os.path.join(path, "generated.wav"), generated, sample_rate)
        return

    @pl.data_loader
    def train_dataloader(self):
        return self.train_loader

    @pl.data_loader
    def val_dataloader(self):
        return self.val_loader
