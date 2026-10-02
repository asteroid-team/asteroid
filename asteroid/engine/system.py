import torch
import pytorch_lightning as pl
import os
import json
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..utils import flatten_dict


class System(pl.LightningModule):
    """Base class for deep learning systems.
    Contains a model, an optimizer, a loss function, training and validation
    dataloaders and learning rate scheduler.

    Note that by default, any PyTorch-Lightning hooks are *not* passed to the model.
    If you want to use Lightning hooks, add the hooks to a subclass::

        class MySystem(System):
            def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
                return self.model.on_train_batch_start(batch, batch_idx, dataloader_idx)

    Args:
        model (torch.nn.Module): Instance of model.
        optimizer (torch.optim.Optimizer): Instance or list of optimizers.
        loss_func (callable): Loss function with signature
            (est_targets, targets).
        train_loader (torch.utils.data.DataLoader): Training dataloader.
        val_loader (torch.utils.data.DataLoader): Validation dataloader.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Instance, or list
            of learning rate schedulers. Also supports dict or list of dict as
            ``{"interval": "step", "scheduler": sched}`` where ``interval=="step"``
            for step-wise schedulers and ``interval=="epoch"`` for classical ones.
        config: Anything to be saved with the checkpoints during training.
            The config dictionary to re-instantiate the run for example.

    .. note:: By default, ``training_step`` (used by ``pytorch-lightning`` in the
        training loop) and ``validation_step`` (used for the validation loop)
        share ``common_step``. If you want different behavior for the training
        loop and the validation loop, overwrite both ``training_step`` and
        ``validation_step`` instead.

    For more info on its methods, properties and hooks, have a look at lightning's docs:
    https://pytorch-lightning.readthedocs.io/en/stable/lightning_module.html#lightningmodule-api
    """

    default_monitor: str = "val_loss"

    def __init__(
        self,
        model,
        optimizer,
        loss_func,
        train_loader,
        val_loader=None,
        scheduler=None,
        config=None,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.config = {} if config is None else config
        self.log_loss_components = bool(self.config.get("loss", {}).get("debug_log_components", False))
        training_conf = self.config.get("training", {})
        self.debug_manifest_batches = int(training_conf.get("debug_manifest_batches", 0) or 0)
        self.debug_manifest_dir = training_conf.get("debug_manifest_dir", "")
        self.debug_manifest_enabled = self.debug_manifest_batches > 0 and bool(self.debug_manifest_dir)
        self._train_identity_keys = set()
        self._val_identity_keys = set()
        # Save lightning's AttributeDict under self.hparams
        self.save_hyperparameters(self.config_to_hparams(self.config))

    def forward(self, *args, **kwargs):
        """Applies forward pass of the model.

        Returns:
            :class:`torch.Tensor`
        """
        return self.model(*args, **kwargs)

    def common_step(self, batch, batch_nb, train=True):
        """Common forward step between training and validation.

        The function of this method is to unpack the data given by the loader,
        forward the batch through the model and compute the loss.
        Pytorch-lightning handles all the rest.

        Args:
            batch: the object returned by the loader (a list of torch.Tensor
                in most cases) but can be something else.
            batch_nb (int): The number of the batch in the epoch.
            train (bool): Whether in training mode. Needed only if the training
                and validation steps are fundamentally different, otherwise,
                pytorch-lightning handles the usual differences.

        Returns:
            :class:`torch.Tensor` : The loss value on this batch.

        .. note::
            This is typically the method to overwrite when subclassing
            ``System``. If the training and validation steps are somehow
            different (except for ``loss.backward()`` and ``optimzer.step()``),
            the argument ``train`` can be used to switch behavior.
            Otherwise, ``training_step`` and ``validation_step`` can be overwriten.
        """
        metadata = None
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            inputs, targets = batch[0], batch[1]
            if len(batch) > 2:
                metadata = batch[2]
        else:
            raise TypeError("Batch must be tuple/list with at least (inputs, targets).")
        est_targets = self(inputs)
        if self.log_loss_components:
            try:
                loss_output = self.loss_func(est_targets, targets, return_components=True)
            except TypeError:
                loss_output = self.loss_func(est_targets, targets)
        else:
            loss_output = self.loss_func(est_targets, targets)

        if isinstance(loss_output, tuple):
            loss, components = loss_output
        else:
            loss = loss_output
            components = {}
        return loss, components, metadata

    def _is_rank_zero(self):
        if self.trainer is None:
            return True
        return int(getattr(self.trainer, "global_rank", 0)) == 0

    def _prepare_manifest_dir(self):
        if not self.debug_manifest_enabled or not self._is_rank_zero():
            return
        os.makedirs(self.debug_manifest_dir, exist_ok=True)

    def _extract_sample_field(self, value, idx):
        if torch.is_tensor(value):
            if value.ndim == 0:
                return value.item()
            if value.shape[0] > idx:
                item = value[idx]
                return item.item() if item.ndim == 0 else item.tolist()
            return None
        if isinstance(value, (list, tuple)):
            if len(value) > idx:
                item = value[idx]
                if torch.is_tensor(item):
                    return item.item() if item.ndim == 0 else item.tolist()
                return item
            return None
        return value

    def _metadata_records(self, metadata, batch_size):
        if not isinstance(metadata, dict):
            return []
        records = []
        for idx in range(batch_size):
            record = {}
            for key, value in metadata.items():
                record[key] = self._extract_sample_field(value, idx)
            records.append(record)
        return records

    def _write_manifest_records(self, split, batch_nb, metadata, batch_size):
        if not self.debug_manifest_enabled:
            return
        if batch_nb >= self.debug_manifest_batches:
            return
        if not self._is_rank_zero():
            return
        if self.trainer is not None and self.trainer.sanity_checking:
            return
        self._prepare_manifest_dir()
        records = self._metadata_records(metadata, batch_size)
        if not records:
            return
        out_path = os.path.join(
            self.debug_manifest_dir,
            f"{split}_manifest_rank0.jsonl",
        )
        with open(out_path, "a", encoding="utf-8") as handle:
            for record in records:
                payload = {
                    "epoch": int(self.current_epoch),
                    "batch_idx": int(batch_nb),
                    "global_step": int(self.global_step),
                    "split": split,
                    "record": record,
                }
                handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def _collect_identity_keys(self, split, metadata, batch_size):
        records = self._metadata_records(metadata, batch_size)
        if not records:
            return
        keys = []
        for record in records:
            key = record.get("identity_key", None)
            if key:
                keys.append(str(key))
        if not keys:
            return
        if split == "train":
            self._train_identity_keys.update(keys)
        else:
            self._val_identity_keys.update(keys)

    def _batch_size_from_batch(self, batch):
        if not isinstance(batch, (tuple, list)) or len(batch) == 0:
            return 0
        inputs = batch[0]
        if torch.is_tensor(inputs):
            return int(inputs.shape[0])
        if hasattr(inputs, "__len__"):
            return int(len(inputs))
        return 0

    def on_train_epoch_start(self):
        self._train_identity_keys = set()

    def on_validation_epoch_start(self):
        self._val_identity_keys = set()

    def _log_components(self, components, prefix, sync_dist):
        for name, value in components.items():
            if not torch.is_tensor(value):
                value = torch.tensor(value, device=self.device, dtype=torch.float32)
            self.log(
                f"{prefix}_{name}",
                value.detach(),
                on_epoch=True,
                prog_bar=False,
                sync_dist=sync_dist,
            )

    def training_step(self, batch, batch_nb):
        """Pass data through the model and compute the loss.

        Backprop is **not** performed (meaning PL will do it for you).

        Args:
            batch: the object returned by the loader (a list of torch.Tensor
                in most cases) but can be something else.
            batch_nb (int): The number of the batch in the epoch.

        Returns:
            torch.Tensor, the value of the loss.
        """
        loss, components, metadata = self.common_step(batch, batch_nb, train=True)
        self.log("loss", loss, logger=True)
        batch_size = self._batch_size_from_batch(batch)
        self._write_manifest_records("train", batch_nb, metadata, batch_size)
        self._collect_identity_keys("train", metadata, batch_size)
        if components:
            self._log_components(
                components,
                "train_loss",
                sync_dist=bool(self.trainer is not None and self.trainer.world_size > 1),
            )
        return loss

    def validation_step(self, batch, batch_nb):
        """Need to overwrite PL validation_step to do validation.

        Args:
            batch: the object returned by the loader (a list of torch.Tensor
                in most cases) but can be something else.
            batch_nb (int): The number of the batch in the epoch.
        """
        loss, components, metadata = self.common_step(batch, batch_nb, train=False)
        batch_size = self._batch_size_from_batch(batch)
        self._write_manifest_records("val", batch_nb, metadata, batch_size)
        self._collect_identity_keys("val", metadata, batch_size)
        # Aggregate validation metric across devices so checkpointing/schedulers
        # use a consistent global value under DDP.
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        if components:
            self._log_components(components, "val_loss", sync_dist=True)

    def on_validation_epoch_end(self):
        """Log hp_metric to tensorboard for hparams selection."""
        if self.trainer is not None and self.trainer.sanity_checking:
            return
        if self.debug_manifest_enabled and self._is_rank_zero():
            overlap = sorted(self._train_identity_keys.intersection(self._val_identity_keys))
            overlap_report_path = os.path.join(
                self.debug_manifest_dir,
                "manifest_overlap_report_rank0.jsonl",
            )
            report = {
                "epoch": int(self.current_epoch),
                "train_unique": len(self._train_identity_keys),
                "val_unique": len(self._val_identity_keys),
                "overlap_count": len(overlap),
                "overlap_examples": overlap[:10],
            }
            with open(overlap_report_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(report, sort_keys=True) + "\n")
            self.print(
                "Manifest overlap epoch "
                f"{self.current_epoch}: overlap_count={report['overlap_count']} "
                f"(train={report['train_unique']}, val={report['val_unique']})"
            )
        hp_metric = self.trainer.callback_metrics.get("val_loss", None)
        if hp_metric is not None:
            self.trainer.logger.log_metrics({"hp_metric": hp_metric}, step=self.trainer.global_step)

    def configure_optimizers(self):
        """Initialize optimizers, batch-wise and epoch-wise schedulers."""
        if self.scheduler is None:
            return self.optimizer

        if not isinstance(self.scheduler, (list, tuple)):
            self.scheduler = [self.scheduler]  # support multiple schedulers

        epoch_schedulers = []
        for sched in self.scheduler:
            if not isinstance(sched, dict):
                if isinstance(sched, ReduceLROnPlateau):
                    sched = {"scheduler": sched, "monitor": self.default_monitor}
                epoch_schedulers.append(sched)
            else:
                sched.setdefault("monitor", self.default_monitor)
                sched.setdefault("frequency", 1)
                # Backward compat
                if sched["interval"] == "batch":
                    sched["interval"] = "step"
                assert sched["interval"] in [
                    "epoch",
                    "step",
                ], "Scheduler interval should be either step or epoch"
                epoch_schedulers.append(sched)
        return [self.optimizer], epoch_schedulers

    def lr_scheduler_step(self, scheduler, metric):
        if metric is None:
            scheduler.step()
        else:
            scheduler.step(metric)

    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return self.val_loader

    def on_save_checkpoint(self, checkpoint):
        """Overwrite if you want to save more things in the checkpoint."""
        checkpoint["training_config"] = self.config
        return checkpoint

    @staticmethod
    def config_to_hparams(dic):
        """Sanitizes the config dict to be handled correctly by torch
        SummaryWriter. It flatten the config dict, converts ``None`` to
        ``"None"`` and any list and tuple into torch.Tensors.

        Args:
            dic (dict): Dictionary to be transformed.

        Returns:
            dict: Transformed dictionary.
        """
        dic = flatten_dict(dic)
        for k, v in dic.items():
            if v is None:
                dic[k] = str(v)
            elif isinstance(v, (list, tuple)):
                dic[k] = torch.tensor(v)
        return dic
