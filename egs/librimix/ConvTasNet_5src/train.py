import os
import argparse
import json
import ast

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from asteroid.models import ConvTasNet
from asteroid.data import VariableLibriMix, OnlineMixDataset
from asteroid.engine.optimizers import make_optimizer
from asteroid.engine.system import System
from asteroid.losses import (
    PITLossWrapper,
    SilenceRobustPairwiseNegSDR,
    ActiveOnlyPITSilencePenalty,
)

parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", default="exp/tmp", help="Full path to save best validation model")


def _parse_speaker_count_weights(weights):
    if weights is None:
        return None
    if isinstance(weights, list):
        if not weights:
            return None
        if all(isinstance(x, str) and len(x) == 1 for x in weights):
            return _parse_speaker_count_weights("".join(weights))
        return [float(x) for x in weights]
    if isinstance(weights, str):
        text = weights.strip()
        if not text:
            return None
        if text.startswith("["):
            parsed = ast.literal_eval(text)
            return [float(x) for x in parsed]
        return [float(x) for x in text.split(",")]
    return None


def main(conf):
    speaker_count_weights = _parse_speaker_count_weights(
        conf["data"].get("speaker_count_weights", None)
    )
    conf["data"]["speaker_count_weights"] = speaker_count_weights
    min_speakers = conf["data"].get("min_speakers", 1)
    max_speakers = conf["data"].get("max_speakers", 5)
    if speaker_count_weights is not None:
        expected = max_speakers - min_speakers + 1
        if len(speaker_count_weights) != expected:
            raise ValueError(
                "speaker_count_weights length mismatch: "
                f"expected {expected} values for counts [{min_speakers}..{max_speakers}], "
                f"got {len(speaker_count_weights)}"
            )
    if conf["data"].get("dataset_type", "variable_librimix") == "online_mix":
        train_set = OnlineMixDataset(
            source_dir=conf["data"]["source_dir"],
            n_src=conf["data"]["n_src"],
            sample_rate=conf["data"]["sample_rate"],
            segment=conf["data"]["segment"],
            num_examples=conf["data"].get("num_examples", 20000),
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            speaker_count_weights=speaker_count_weights,
        )
        val_set = OnlineMixDataset(
            source_dir=conf["data"]["valid_source_dir"],
            n_src=conf["data"]["n_src"],
            sample_rate=conf["data"]["sample_rate"],
            segment=conf["data"]["segment"],
            num_examples=conf["data"].get("val_num_examples", 2000),
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            speaker_count_weights=speaker_count_weights,
            seed=42,
        )
    else:
        train_set = VariableLibriMix(
            csv_dirs=conf["data"]["train_dir"],
            task=conf["data"]["task"],
            sample_rate=conf["data"]["sample_rate"],
            n_src=conf["data"]["n_src"],
            segment=conf["data"]["segment"],
        )

        val_set = VariableLibriMix(
            csv_dirs=conf["data"]["valid_dir"],
            task=conf["data"]["task"],
            sample_rate=conf["data"]["sample_rate"],
            n_src=conf["data"]["n_src"],
            segment=conf["data"]["segment"],
        )

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
    )
    conf["masknet"].update({"n_src": conf["data"]["n_src"]})

    model = ConvTasNet(
        **conf["filterbank"], **conf["masknet"], sample_rate=conf["data"]["sample_rate"]
    )
    optimizer = make_optimizer(model.parameters(), **conf["optim"])
    # Define scheduler
    scheduler = None
    if conf["training"]["half_lr"]:
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=5)
    # Just after instantiating, save the args. Easy loading in the future.
    exp_dir = conf["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    # Define Loss function.
    loss_conf = conf.get("loss", {})
    silence_threshold = loss_conf.get("silence_threshold", 1e-5)
    loss_mode = loss_conf.get("mode", "active_only_pit")
    if loss_mode == "legacy_silence_robust":
        loss_func = PITLossWrapper(
            SilenceRobustPairwiseNegSDR("sisdr", threshold=silence_threshold),
            pit_from="pw_mtx",
        )
    elif loss_mode == "active_only_pit":
        loss_func = ActiveOnlyPITSilencePenalty(
            threshold=silence_threshold,
            silence_margin_db=loss_conf.get("silence_margin_db", -45.0),
            silence_metric=loss_conf.get("silence_metric", "rms_db"),
            active_weight=loss_conf.get("active_weight", 1.0),
            silence_weight=loss_conf.get("silence_weight", 0.2),
            active_metric=loss_conf.get("active_metric", "sisdr"),
            active_threshold_mode=loss_conf.get("active_threshold_mode", "rms"),
            active_l1_weight=loss_conf.get("active_l1_weight", 0.0),
            active_l2_weight=loss_conf.get("active_l2_weight", 0.0),
            active_rms_floor_db=loss_conf.get("active_rms_floor_db", None),
            active_rms_penalty_weight=loss_conf.get("active_rms_penalty_weight", 0.0),
        )
    else:
        raise ValueError(
            f"Unknown loss.mode={loss_mode}. "
            "Supported: active_only_pit, legacy_silence_robust."
        )
    system = System(
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

    use_gpu = torch.cuda.is_available()
    n_devices = torch.cuda.device_count() if use_gpu else 1
    trainer_strategy = "ddp_find_unused_parameters_true" if n_devices > 1 else "auto"

    trainer = pl.Trainer(
        max_epochs=conf["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        accelerator="gpu" if use_gpu else "cpu",
        strategy=trainer_strategy,
        devices=n_devices if use_gpu else 1,
        limit_train_batches=1.0,
        gradient_clip_val=5.0,
    )
    trainer.fit(system)

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.model.serialize()
    to_save.update(train_set.get_infos())
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))


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
