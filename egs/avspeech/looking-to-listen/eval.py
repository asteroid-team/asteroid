import sys
import yaml
import collections
from pathlib import Path
from argparse import ArgumentParser
import torch
import numpy as np
from catalyst.dl import utils
from catalyst.dl.runner import SupervisedRunner
from asteroid.data.avspeech_dataset import AVSpeechDataset

from local.loader.constants import EMBED_DIR
from model import make_model_and_optimizer, load_best_model
from train import SNRCallback, SDRCallback, ParamConfig


def validate(model, val_dataset, config):
    loaders = collections.OrderedDict()
    val_loader = utils.get_loader(
        val_dataset,
        open_fn=lambda x: {"input_audio": x[-1], "input_video": x[1], "targets": x[0]},
        batch_size=config.batch_size,
        num_workers=config.workers,
        shuffle=False,
    )

    loaders["valid"] = val_loader

    runner = SupervisedRunner(
        input_key=["input_audio", "input_video"]
    )  # parameters of the model in forward(...)
    runner.infer(
        model,
        loaders,
        callbacks=collections.OrderedDict(
            {"snr_callback": SNRCallback(), "sdr_callback": SDRCallback()}
        ),
        verbose=True,
    )


def main(conf):
    config = ParamConfig(
        conf["training"]["batch_size"],
        conf["training"]["epochs"],
        conf["training"]["num_workers"],
        cuda=True,
        use_half=False,
        learning_rate=conf["optim"]["lr"],
    )

    val_dataset = AVSpeechDataset(Path("data/val.csv"), Path(EMBED_DIR), conf["main_args"]["n_src"])

    model = load_best_model(conf, conf["main_args"]["exp_dir"])

    print(f"AVFusion has {sum(np.prod(i.shape) for i in model.parameters()):,} parameters")

    if torch.cuda.device_count() > 1:
        print(f"Multiple GPUs available")
        device_ids = (
            list(map(int, conf["main_args"]["gpus"].split(",")))
            if conf["main_args"]["gpus"] != "-1"
            else None
        )
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    validate(model, val_dataset, config)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpus", type=str, help="list of GPUs", default="-1")
    parser.add_argument(
        "--n-src",
        type=int,
        help="number of inputs to neural network",
        default=2,
    )
    parser.add_argument(
        "--exp_dir",
        default="exp/logdir",
        help="Full path to save best validation model",
    )

    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    # We start with opening the config file conf.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    with open("local/conf.yml") as f:
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
