from asteroid.engine.system import System as SystemCore


class SystemTwoStep(SystemCore):
    """
    Inherits from the core system class and overrides the methods for the
    common steps as well the train and evaluation steps for the two-step
    source separation.

    Args:
        model (torch.nn.Module): Instance of model.
        optimizer (torch.optim.Optimizer): Instance or list of optimizers.
        loss_func (callable): Loss function with signature
            (est_targets, targets).
        train_loader (torch.utils.data.DataLoader): Training dataloader.
        val_loader (torch.utils.data.DataLoader): Validation dataloader.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Instance, or list
            of learning rate schedulers.
        config: Anything to be saved with the checkpoints during training.
            The config dictionary to re-instantiate the run for example.
        module (str):
            'separator': The two step is used for training or evaluation the
                         separation module only.
            'filterbank': The two step approach is used for training only the
                          adaptive encoder/decoder part or in other words the
                          filterbank.
            For more info take a look at method common_step_two_step_separtion()
    .. note:: By default, `training_step` (used by `pytorch-lightning` in the
        training loop) and `validation_step` (used for the validation loop)
        share `common_step`. If you want different behavior for the training
        loop and the validation loop, overwrite both `training_step` and
        `validation_step` instead.
    """

    def __init__(
        self,
        model,
        optimizer,
        loss_func,
        train_loader,
        val_loader=None,
        scheduler=None,
        config=None,
        module=None,
    ):
        super().__init__(
            model,
            optimizer,
            loss_func,
            train_loader,
            val_loader=val_loader,
            scheduler=scheduler,
            config=config,
        )
        assert module in ["filterbank", "separator"], (
            "If the two-step  "
            "approach is used then either filterbank or separator has "
            "to be used but got: {}".format(module)
        )
        self.module = module

    def common_step(self, batch, train=True):
        """Common forward step between training and validation.

        The function of this method is to unpack the data given by the loader,
        forward the batch through the model and compute the loss for the
        separation module and the filterbank when the optimization process
        for source separation is breaken in two distinct processes as
        proposed in [1].

        Args:
            batch: the object returned by the loader (a list of torch.Tensor
                in most cases) but can be something else.
            train (bool): In case of training or validation the
                filterbank will return the estimated time signals. However,
                in training mode the separator will be trained using the
                ideal latent targets and it will estimate the corresponding
                latent representations of the sources as proposed in [1].
        Returns:
            dict:

            ``'loss'``: loss

            ``'log'``: dict with tensorboard logs

        References:
            [1]: Tzinis, E., Venkataramani, S., Wang, Z., Subakan, Y. C., and
                 Smaragdis, P., "Two-Step Sound Source Separation:
                 Training on Learned Latent Targets." In Acoustics, Speech
                 and Signal Processing (ICASSP), 2020 IEEE International
                 Conference. https://arxiv.org/abs/1910.09804
        """
        mixture_time, true_sources_time = batch
        if self.module == "filterbank":
            est_sources_time, _ = self(mixture_time, true_sources_time)
            est_sources_time_mean = est_sources_time.mean(-1, keepdims=True)
            true_sources_time_mean = true_sources_time.mean(-1, keepdims=True)
            return self.loss_func(
                est_sources_time - est_sources_time_mean, true_sources_time - true_sources_time_mean
            )

        # Here we train or validate the separator. In training we need the
        # latent targets to regress on. In validation we just provide the
        # estimated time domain signals.
        if train:
            latent_targets = self.model.get_ideal_latent_targets(mixture_time, true_sources_time)
            est_latents = self.model.estimate_latent_representations(mixture_time)
            batch_size, n_sources = est_latents.shape[0], est_latents.shape[1]
            # See section 2.2 of the paper
            return self.loss_func(
                est_latents.view(batch_size, n_sources, -1),
                latent_targets.view(batch_size, n_sources, -1),
            )
        else:
            est_sources_time = self.model(mixture_time)
            est_sources_time_mean = est_sources_time.mean(-1, keepdims=True)
            true_sources_time_mean = true_sources_time.mean(-1, keepdims=True)
            return self.loss_func(
                est_sources_time - est_sources_time_mean, true_sources_time - true_sources_time_mean
            )

    def training_step(self, batch, batch_nb):
        loss = self.common_step(batch, train=True)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        loss = self.common_step(batch, train=False)
        return {"val_loss": loss}
