import os
import argparse
import json
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from asteroid.engine.system import System
from asteroid.losses import PITLossWrapper, pairwise_mse
from asteroid.losses import deep_clustering_loss
from asteroid_filterbanks.transforms import mag
from asteroid.dsp.vad import ebased_vad

from asteroid.data.kinect_wsj import make_dataloaders
from model import make_model_and_optimizer

parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", default="exp/tmp", help="Full path to save best validation model")


def main(conf):
    exp_dir = conf["main_args"]["exp_dir"]
    # Define Dataloader
    train_loader, val_loader = make_dataloaders(**conf["data"], **conf["training"])
    conf["masknet"].update({"n_src": conf["data"]["n_src"]})
    # Define model, optimizer + scheduler
    model, optimizer = make_model_and_optimizer(conf)
    scheduler = None
    if conf["training"]["half_lr"]:
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=5)

    # Save config
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    # Define loss function
    loss_func = ChimeraLoss(alpha=conf["training"]["loss_alpha"])
    # Put together in System
    system = ChimeraSystem(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        config=conf,
    )

    # Define callbacks
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        checkpoint_dir, monitor="val_loss", mode="min", save_top_k=5, verbose=True
    )
    callbacks.append(checkpoint)
    if conf["training"]["early_stop"]:
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=30, verbose=True))

    # Train model
    trainer = pl.Trainer(
        max_epochs=conf["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy="ddp",
        devices="auto",
        limit_train_batches=1.0,  # Useful for fast experiment
        gradient_clip_val=200,
    )
    trainer.fit(system)

    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(checkpoint.best_k_models, f, indent=0)
    # Save last model for convenience
    torch.save(system.model.state_dict(), os.path.join(exp_dir, "final_model.pth"))


# TODO:Should ideally be inherited from wsj0-mix
class ChimeraSystem(System):
    def __init__(self, *args, mask_mixture=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_mixture = mask_mixture

    def common_step(self, batch, batch_nb, train=False):
        inputs, targets, masks = self.unpack_data(batch)
        embeddings, est_masks = self(inputs)
        spec = mag(self.model.encoder(inputs.unsqueeze(1)))
        if self.mask_mixture:
            est_masks = est_masks * spec.unsqueeze(1)
            masks = masks * spec.unsqueeze(1)
        loss, loss_dic = self.loss_func(
            embeddings, targets, est_src=est_masks, target_src=masks, mix_spec=spec
        )
        return loss, loss_dic

    def training_step(self, batch, batch_nb):
        loss, loss_dic = self.common_step(batch, batch_nb, train=True)
        tensorboard_logs = dict(
            train_loss=loss, train_dc_loss=loss_dic["dc_loss"], train_pit_loss=loss_dic["pit_loss"]
        )
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        loss, loss_dic = self.common_step(batch, batch_nb, train=False)
        tensorboard_logs = dict(
            val_loss=loss, val_dc_loss=loss_dic["dc_loss"], val_pit_loss=loss_dic["pit_loss"]
        )
        return {"val_loss": loss, "log": tensorboard_logs}

    def validation_end(self, outputs):
        # Not so pretty for now but it helps.
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_dc_loss = torch.stack([x["log"]["val_dc_loss"] for x in outputs]).mean()
        avg_pit_loss = torch.stack([x["log"]["val_pit_loss"] for x in outputs]).mean()
        tensorboard_logs = dict(
            val_loss=avg_loss, val_dc_loss=avg_dc_loss, val_pit_loss=avg_pit_loss
        )
        return {
            "val_loss": avg_loss,
            "log": tensorboard_logs,
            "progress_bar": {"val_loss": avg_loss},
        }

    def unpack_data(self, batch, EPS=1e-8):
        mix, sources, noise = batch
        # Take only the first channel
        mix = mix[..., 0]
        sources = sources[..., 0]
        noise = noise[..., 0]
        noise = noise.unsqueeze(1)
        # Compute magnitude spectrograms and IRM
        src_mag_spec = mag(self.model.encoder(sources))
        noise_mag_spec = mag(self.model.encoder(noise))
        noise_mag_spec = noise_mag_spec.unsqueeze(1)
        real_mask = src_mag_spec / (noise_mag_spec + src_mag_spec.sum(1, keepdim=True) + EPS)
        # Get the src idx having the maximum energy
        binary_mask = real_mask.argmax(1)
        return mix, binary_mask, real_mask


class ChimeraLoss(nn.Module):
    """Combines Deep clustering loss and mask inference loss for ChimeraNet.

    Args:
        alpha (float): loss weight. Total loss will be :
            `alpha` * dc_loss + (1 - `alpha`) * mask_mse_loss.
    """

    def __init__(self, alpha=0.1):
        super().__init__()
        assert alpha >= 0, "Negative alpha values don't make sense."
        assert alpha <= 1, "Alpha values above 1 don't make sense."
        # PIT loss
        self.src_mse = PITLossWrapper(pairwise_mse, pit_from="pw_mtx")
        self.alpha = alpha

    def forward(self, est_embeddings, target_indices, est_src=None, target_src=None, mix_spec=None):
        """

        Args:
            est_embeddings (torch.Tensor): Estimated embedding from the DC head.
            target_indices (torch.Tensor): Target indices that'll be passed to
                the DC loss.
            est_src (torch.Tensor): Estimated magnitude spectrograms (or masks).
            target_src (torch.Tensor): Target magnitude spectrograms (or masks).
            mix_spec (torch.Tensor): The magnitude spectrogram of the mixture
                from which VAD will be computed. If None, no VAD is used.

        Returns:
            torch.Tensor, the total loss, averaged over the batch.
            dict with `dc_loss` and `pit_loss` keys, unweighted losses.
        """
        if self.alpha != 0 and (est_src is None or target_src is None):
            raise ValueError(
                "Expected target and estimated spectrograms to " "compute the PIT loss, found None."
            )
        binary_mask = None
        if mix_spec is not None:
            binary_mask = ebased_vad(mix_spec)
        # Dc loss is already divided by VAD in the loss function.
        dc_loss = deep_clustering_loss(
            embedding=est_embeddings, tgt_index=target_indices, binary_mask=binary_mask
        )
        src_pit_loss = self.src_mse(est_src, target_src)
        # Equation (4) from Chimera paper.
        tot = self.alpha * dc_loss.mean() + (1 - self.alpha) * src_pit_loss
        # Return unweighted losses as well for logging.
        loss_dict = dict(dc_loss=dc_loss.mean(), pit_loss=src_pit_loss)
        return tot, loss_dict


if __name__ == "__main__":
    import yaml
    from pprint import pprint
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    with open("local/conf.yml") as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    pprint(arg_dic)
    main(arg_dic)
