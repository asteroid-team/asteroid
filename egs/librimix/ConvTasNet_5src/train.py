import os
import argparse
import json
import ast
import random
import math
import csv

import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger, CSVLogger

from asteroid.models import ConvTasNet
from asteroid.data import VariableLibriMix, OnlineMixDataset
from asteroid.engine.optimizers import make_optimizer
from asteroid.engine.system import System
from asteroid.losses import (
    PITLossWrapper,
    SilenceRobustPairwiseNegSDR,
    ActiveOnlyPITSilencePenalty,
    pairwise_neg_sisdr,
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


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _estimate_steps_per_epoch(dataset_len, batch_size, num_devices):
    samples_per_rank = int(math.ceil(dataset_len / max(num_devices, 1)))
    return int(samples_per_rank // batch_size)


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    raise ValueError(f"Cannot parse boolean value from: {value!r}")


def _parse_tags(tags):
    if tags is None:
        return []
    if isinstance(tags, list):
        # `prepare_parser_from_dict` can parse comma-separated CLI strings
        # as a list of single characters for list-typed config fields.
        if tags and all(isinstance(tag, str) and len(tag) == 1 for tag in tags):
            tags = "".join(tags)
        else:
            return [str(tag).strip() for tag in tags if str(tag).strip()]
    if isinstance(tags, str):
        text = tags.strip()
    else:
        text = str(tags).strip()
    if not text:
        return []
    if text.startswith("["):
        parsed = ast.literal_eval(text)
        return [str(tag).strip() for tag in parsed if str(tag).strip()]
    return [tag.strip() for tag in text.split(",") if tag.strip()]


def _build_wandb_logger(training_conf, conf, exp_dir):
    use_wandb = _as_bool(training_conf.get("use_wandb", True))
    if not use_wandb:
        return True

    try:
        __import__("wandb")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "W&B logging is enabled (`training.use_wandb=true`) but `wandb` is not installed. "
            "Install with `pip install wandb` and authenticate with `wandb login` "
            "(or set `WANDB_API_KEY`)."
        ) from exc

    run_name = training_conf.get("wandb_run_name") or os.path.basename(os.path.normpath(exp_dir))
    run_group = training_conf.get("wandb_group")
    if not run_group:
        run_group = os.environ.get("WANDB_GROUP")
    if not run_group:
        run_group = os.path.basename(os.path.dirname(os.path.normpath(exp_dir))) or "manual"

    project = training_conf.get("wandb_project", "librimix-convtasnet-5src")
    entity = training_conf.get("wandb_entity", None)
    tags = _parse_tags(training_conf.get("wandb_tags", []))
    job_type = training_conf.get("wandb_job_type", "train")

    try:
        logger = WandbLogger(
            project=project,
            entity=entity,
            name=run_name,
            group=run_group,
            tags=tags,
            job_type=job_type,
            save_dir=exp_dir,
            log_model=False,
        )
        # Force backend initialization/auth check now so failures happen before training.
        _ = logger.experiment
    except Exception as exc:
        raise RuntimeError(
            "W&B logging is enabled but initialization failed. "
            "Run `wandb login` (or set `WANDB_API_KEY`) and verify project/entity settings."
        ) from exc

    # W&B/Lightning versions differ on how run config is exposed.
    # Try common update paths, but never crash training on config-sync only.
    try:
        run = logger.experiment
        run_config = getattr(run, "config", None)
        if hasattr(run_config, "update"):
            run_config.update(conf, allow_val_change=True)
        else:
            import wandb

            if hasattr(wandb, "config") and hasattr(wandb.config, "update"):
                wandb.config.update(conf, allow_val_change=True)
    except Exception as exc:
        print(f"W&B config sync warning: {exc}")
    return logger


def _build_loggers(training_conf, conf, exp_dir):
    loggers = []
    use_wandb = _as_bool(training_conf.get("use_wandb", True))
    if use_wandb:
        loggers.append(_build_wandb_logger(training_conf, conf, exp_dir))
    # Always keep a local CSV logger so metric plotting is available.
    loggers.append(CSVLogger(save_dir=exp_dir, name="lightning_csv"))
    if len(loggers) == 1:
        return loggers[0]
    return loggers


def _read_metric_series(metrics_csv_path):
    if not os.path.isfile(metrics_csv_path):
        return {}
    series = {}
    with open(metrics_csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            step_text = row.get("step", "")
            epoch_text = row.get("epoch", "")
            step = int(float(step_text)) if step_text not in ("", None) else None
            epoch = int(float(epoch_text)) if epoch_text not in ("", None) else None
            for key, value in row.items():
                if key in ("step", "epoch") or value in ("", None):
                    continue
                try:
                    metric_value = float(value)
                except ValueError:
                    continue
                if key not in series:
                    series[key] = {"step": [], "epoch": [], "value": []}
                series[key]["step"].append(step)
                series[key]["epoch"].append(epoch)
                series[key]["value"].append(metric_value)
    return series


def _save_metric_plots(training_conf, metrics_csv_path, plot_dir):
    if not _as_bool(training_conf.get("plot_metrics", True)):
        return
    series = _read_metric_series(metrics_csv_path)
    if not series:
        print(f"No plottable metrics found in {metrics_csv_path}")
        return
    prefixes = training_conf.get("plot_metric_prefixes", ["val_", "test_"])
    if isinstance(prefixes, str):
        prefixes = [item.strip() for item in prefixes.split(",") if item.strip()]
    if not prefixes:
        prefixes = ["val_", "test_"]

    selected_keys = []
    for key in sorted(series.keys()):
        if any(key.startswith(prefix) for prefix in prefixes):
            selected_keys.append(key)
    if not selected_keys:
        print(
            "No metrics matched configured plotting prefixes "
            f"{prefixes}; skipping metric plots."
        )
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping metric plotting because matplotlib is unavailable: {exc}")
        return

    os.makedirs(plot_dir, exist_ok=True)

    for key in selected_keys:
        values = series[key]["value"]
        steps = series[key]["step"]
        epochs = series[key]["epoch"]
        x_values = steps if all(step is not None for step in steps) else list(range(len(values)))
        x_label = "Global Step" if all(step is not None for step in steps) else "Index"
        if all(epoch is not None for epoch in epochs):
            x_values = epochs
            x_label = "Epoch"

        fig = plt.figure(figsize=(8, 4.5))
        plt.plot(x_values, values, marker="o", linewidth=1.8)
        plt.title(key)
        plt.xlabel(x_label)
        plt.ylabel(key)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(plot_dir, f"{key}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    # Combined summary figure for quick overview
    fig = plt.figure(figsize=(10, 5.5))
    for key in selected_keys:
        values = series[key]["value"]
        epochs = series[key]["epoch"]
        steps = series[key]["step"]
        x_values = epochs if all(epoch is not None for epoch in epochs) else steps
        if not all(v is not None for v in x_values):
            x_values = list(range(len(values)))
        plt.plot(x_values, values, linewidth=1.4, label=key)
    plt.title("Validation/Test Metrics Over Time")
    plt.xlabel("Epoch/Step")
    plt.ylabel("Metric Value")
    plt.grid(alpha=0.3)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    combined_path = os.path.join(plot_dir, "validation_test_metrics_over_time.png")
    fig.savefig(combined_path, dpi=160)
    plt.close(fig)
    print(f"Saved metric plots to {plot_dir}")


def _latest_metrics_csv(exp_dir):
    base_dir = os.path.join(exp_dir, "lightning_csv")
    if not os.path.isdir(base_dir):
        return None
    version_dirs = []
    for name in os.listdir(base_dir):
        if not name.startswith("version_"):
            continue
        suffix = name.split("_", 1)[-1]
        if suffix.isdigit():
            version_dirs.append((int(suffix), os.path.join(base_dir, name)))
    if not version_dirs:
        return None
    version_dirs.sort(key=lambda item: item[0], reverse=True)
    for _, version_dir in version_dirs:
        candidate = os.path.join(version_dir, "metrics.csv")
        if os.path.isfile(candidate):
            return candidate
    return None


def main(conf):
    training_conf = conf.get("training", {})
    data_conf = conf.get("data", {})
    # Normalize tags early and store as CSV string so Asteroid hparams tensor
    # conversion doesn't crash on list[str].
    normalized_tags = _parse_tags(training_conf.get("wandb_tags", []))
    training_conf["wandb_tags"] = ",".join(normalized_tags)
    seed = int(training_conf.get("seed", 1337))
    pl.seed_everything(seed, workers=True)

    speaker_count_weights = _parse_speaker_count_weights(
        data_conf.get("speaker_count_weights", None)
    )
    conf["data"]["speaker_count_weights"] = speaker_count_weights
    min_speakers = data_conf.get("min_speakers", 1)
    max_speakers = data_conf.get("max_speakers", 5)
    debug_manifest_batches = int(training_conf.get("debug_manifest_batches", 0) or 0)
    debug_manifest_dir = training_conf.get("debug_manifest_dir", "")
    debug_hash_audio = _as_bool(training_conf.get("debug_hash_audio", False))
    return_metadata = debug_manifest_batches > 0 and bool(debug_manifest_dir)

    if speaker_count_weights is not None:
        expected = max_speakers - min_speakers + 1
        if len(speaker_count_weights) != expected:
            raise ValueError(
                "speaker_count_weights length mismatch: "
                f"expected {expected} values for counts [{min_speakers}..{max_speakers}], "
                f"got {len(speaker_count_weights)}"
            )
    if data_conf.get("dataset_type", "variable_librimix") == "online_mix":
        train_set = OnlineMixDataset(
            source_dir=data_conf["source_dir"],
            n_src=data_conf["n_src"],
            sample_rate=data_conf["sample_rate"],
            segment=data_conf["segment"],
            num_examples=data_conf.get("num_examples", 20000),
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            speaker_count_weights=speaker_count_weights,
            seed=data_conf.get("train_seed", None),
            return_metadata=return_metadata,
            hash_audio=debug_hash_audio,
        )
        val_set = OnlineMixDataset(
            source_dir=data_conf["valid_source_dir"],
            n_src=data_conf["n_src"],
            sample_rate=data_conf["sample_rate"],
            segment=data_conf["segment"],
            num_examples=data_conf.get("val_num_examples", 2000),
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            speaker_count_weights=speaker_count_weights,
            seed=data_conf.get("val_seed", 42),
            return_metadata=return_metadata,
            hash_audio=debug_hash_audio,
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

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    val_generator = torch.Generator()
    val_generator.manual_seed(seed + 1)

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=training_conf["batch_size"],
        num_workers=training_conf["num_workers"],
        drop_last=True,
        worker_init_fn=_seed_worker,
        generator=train_generator,
    )

    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=training_conf["batch_size"],
        num_workers=training_conf["num_workers"],
        drop_last=True,
        worker_init_fn=_seed_worker,
        generator=val_generator,
    )
    conf["masknet"].update({"n_src": data_conf["n_src"]})

    model = ConvTasNet(
        **conf["filterbank"], **conf["masknet"], sample_rate=data_conf["sample_rate"]
    )
    optimizer = make_optimizer(model.parameters(), **conf["optim"])
    # Define scheduler
    scheduler = None
    if training_conf["half_lr"]:
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
    elif loss_mode == "standard_pit_sisdr":
        loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
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
            "Supported: active_only_pit, legacy_silence_robust, standard_pit_sisdr."
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
    if training_conf["early_stop"]:
        callbacks.append(
            EarlyStopping(
                monitor=training_conf.get("early_stop_monitor", "val_loss"),
                mode=training_conf.get("early_stop_mode", "min"),
                patience=int(training_conf.get("early_stop_patience", 30)),
                min_delta=float(training_conf.get("early_stop_min_delta", 0.0)),
                verbose=True,
            )
        )

    use_gpu = torch.cuda.is_available()
    n_devices = torch.cuda.device_count() if use_gpu else 1
    trainer_strategy = "ddp_find_unused_parameters_true" if n_devices > 1 else "auto"
    deterministic_debug = _as_bool(training_conf.get("deterministic_debug", False))
    if deterministic_debug:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    train_steps = _estimate_steps_per_epoch(
        len(train_set), training_conf["batch_size"], n_devices if use_gpu else 1
    )
    val_steps = _estimate_steps_per_epoch(
        len(val_set), training_conf["batch_size"], n_devices if use_gpu else 1
    )
    print(
        "Effective loader stats: "
        f"train_examples={len(train_set)}, val_examples={len(val_set)}, "
        f"batch_size={training_conf['batch_size']}, devices={(n_devices if use_gpu else 1)}, "
        f"train_steps_per_epoch~{train_steps}, val_steps_per_epoch~{val_steps}"
    )
    trainer_logger = _build_loggers(training_conf, conf, exp_dir)
    if isinstance(trainer_logger, WandbLogger) and _as_bool(
        training_conf.get("wandb_watch_model", False)
    ):
        trainer_logger.watch(model, log="all", log_freq=100)
    elif isinstance(trainer_logger, list):
        for logger in trainer_logger:
            if isinstance(logger, WandbLogger) and _as_bool(
                training_conf.get("wandb_watch_model", False)
            ):
                logger.watch(model, log="all", log_freq=100)

    trainer = pl.Trainer(
        max_epochs=training_conf["epochs"],
        callbacks=callbacks,
        logger=trainer_logger,
        default_root_dir=exp_dir,
        accelerator="gpu" if use_gpu else "cpu",
        strategy=trainer_strategy,
        devices=n_devices if use_gpu else 1,
        limit_train_batches=1.0,
        gradient_clip_val=5.0,
        deterministic=deterministic_debug,
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
    if getattr(trainer, "is_global_zero", True):
        csv_metrics_path = _latest_metrics_csv(exp_dir)
        plot_dir = os.path.join(exp_dir, "metric_plots")
        if csv_metrics_path is None:
            print("No CSV metrics file found; skipping metric plotting.")
        else:
            _save_metric_plots(training_conf, csv_metrics_path, plot_dir)

    if isinstance(trainer_logger, WandbLogger) and getattr(trainer, "is_global_zero", True):
        trainer_logger.experiment.finish()
    elif isinstance(trainer_logger, list) and getattr(trainer, "is_global_zero", True):
        for logger in trainer_logger:
            if isinstance(logger, WandbLogger):
                logger.experiment.finish()


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
