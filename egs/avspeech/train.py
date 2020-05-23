import sys
import yaml

import torch
import numpy as np
from pathlib import Path
from memory_profiler import profile
from argparse import ArgumentParser

from asteroid.data import AVSpeechDataset

from local import Signal, convert_to_spectrogram
from train import train, ParamConfig
from model import (make_model_and_optimizer,
                   load_best_model)

class DiscriminativeLoss(torch.nn.Module):
    #Reference: https://github.com/bill9800/speech_separation/blob/master/model/lib/model_loss.py

    def __init__(self, input_audio_size=2, gamma=0.1):
        super(DiscriminativeLoss, self).__init__()

        self.input_audio_size = input_audio_size
        self.gamma = gamma

    def forward(self, input, target):

        sum_mtr = torch.zeros_like(input[..., 0])
        for i in range(self.input_audio_size):
            sum_mtr += ((target[:,:,:,:,i]-input[:,:,:,:,i]) ** 2)
            for j in range(self.input_audio_size):
                if i != j:
                    sum_mtr -= (self.gamma * ((target[:,:,:,:,i]-input[:,:,:,:,j]) ** 2))
        sum_mtr = torch.mean(sum_mtr.view(-1))

        return sum_mtr


def main(conf):
    config = ParamConfig(conf["training"]["batch_size"], conf["training"]["epochs"],
                         conf["training"]["num_workers"], True, False, conf["optim"]["lr"])
    dataset = AVSpeechDataset(Path("data/train.csv"), Signal, convert_to_spectrogram, conf["data"]["input_audio_size"])
    val_dataset = AVSpeechDataset(Path("data/val.csv"), Signal, convert_to_spectrogram, conf["data"]["input_audio_size"])

    model, optimizer = make_model_and_optimizer(conf)
    print(f"AVFusion has {sum(np.prod(i.shape) for i in model.parameters()):,} parameters")

    criterion = DiscriminativeLoss()

    #if args.model_path and args.model_path.is_file():
    #    resume = args.model_path.as_posix()
    #else:
    resume = None

    train(model, dataset, optimizer, criterion, config, val_dataset=val_dataset, resume=resume,
          logdir=conf["training"]["logdir"])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--gpus', type=str, help='list of GPUs', default='-1')
    parser.add_argument('--exp_dir', default='exp/logdir',
                        help='Full path to save best validation model')

    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    # We start with opening the config file conf.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    with open('local/conf.yml') as f:
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

    """
    parser = ArgumentParser()
    parser.add_argument("--bs", default=2, type=int, help="batch size of dataset")
    parser.add_argument("--epochs", default=40, type=int, help="max epochs to train")
    parser.add_argument("--cuda", default=True, type=bool, help="cuda for training")
    parser.add_argument("--workers", default=0, type=int, help="total workers for dataset")
    parser.add_argument("--input-audio-size", default=2, type=int, help="total input size")
    parser.add_argument("--dataset-path", default=Path("../data/audio_visual/avspeech_train.csv"), type=Path, help="path for avspeech training data")
    parser.add_argument("--video-dir", default=Path("../data/train"), type=Path, help="directory where all videos are stored")
    parser.add_argument("--input-df-path", default=Path("train.csv"), type=Path, help="path for combinations dataset")
    parser.add_argument("--val-input-df-path", default=Path("val.csv"), type=Path, help="path for combinations dataset")
    parser.add_argument("--use-half", default=False, type=bool, help="halves the precision")
    parser.add_argument("--learning-rate", default=3e-4, type=float, help="learning rate for the network")
    parser.add_argument("--model-path", default=Path("logdir/checkpoints/best_full.pth"), type=Path, help="Partially trained model path")

    args = parser.parse_args()

    main(args)
    """