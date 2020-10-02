import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from asteroid.data.avspeech_dataset import AVSpeechDataset

from local.loader.constants import EMBED_DIR
from train import train, ParamConfig
from model import make_model_and_optimizer, load_best_model


class DiscriminativeLoss(torch.nn.Module):
    # Reference: https://github.com/bill9800/speech_separation/blob/master/model/lib/model_loss.py

    def __init__(self, n_src=2, gamma=0.1):
        super(DiscriminativeLoss, self).__init__()

        self.n_src = n_src
        self.gamma = gamma

    def forward(self, input, target):

        sum_mtr = torch.zeros_like(input[:, 0, ...])
        for i in range(self.n_src):
            sum_mtr += (target[:, i, ...] - input[:, i, ...]) ** 2
            for j in range(self.n_src):
                if i != j:
                    sum_mtr -= self.gamma * ((target[:, i, ...] - input[:, j, ...]) ** 2)
        sum_mtr = torch.mean(sum_mtr.view(-1))

        return sum_mtr


def main(conf):
    config = ParamConfig(
        conf["training"]["batch_size"],
        conf["training"]["epochs"],
        conf["training"]["num_workers"],
        cuda=True,
        use_half=False,
        learning_rate=conf["optim"]["lr"],
    )

    dataset = AVSpeechDataset(Path("data/train.csv"), Path(EMBED_DIR), conf["main_args"]["n_src"])
    val_dataset = AVSpeechDataset(Path("data/val.csv"), Path(EMBED_DIR), conf["main_args"]["n_src"])

    model, optimizer = make_model_and_optimizer(conf)
    print(f"AVFusion has {sum(np.prod(i.shape) for i in model.parameters()):,} parameters")

    criterion = DiscriminativeLoss()

    model_path = Path(conf["main_args"]["exp_dir"]) / "checkpoints" / "best_full.pth"
    if model_path.is_file():
        print("Loading saved model...")
        resume = model_path.as_posix()
    else:
        resume = None

    if torch.cuda.device_count() > 1:
        print(f"Multiple GPUs available")
        device_ids = (
            list(map(int, conf["main_args"]["gpus"].split(",")))
            if conf["main_args"]["gpus"] != "-1"
            else None
        )
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    train(
        model,
        dataset,
        optimizer,
        criterion,
        config,
        val_dataset=val_dataset,
        resume=resume,
        logdir=conf["main_args"]["exp_dir"],
    )


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
