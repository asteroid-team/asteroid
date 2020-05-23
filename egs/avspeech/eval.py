import sys

import collections
from pathlib import Path
from argparse import ArgumentParser

import torch
import numpy as np
from catalyst.dl import utils
from catalyst.dl.runner import SupervisedRunner

from asteroid.data import AVSpeechDataset

from model import (make_model_and_optimizer,
                   load_best_model)
from train import SNRCallback, SDRCallback, ParamConfig

def validate(model, dataset, val_dataset, config):
    loaders = collections.OrderedDict()
    train_loader = utils.get_loader(dataset, open_fn=lambda x: {"input_audio": x[-1], "input_video": x[1], "targets": x[0]},
                                    batch_size=config.batch_size, num_workers=config.workers, shuffle=True)

    val_loader = utils.get_loader(val_dataset, open_fn=lambda x: {"input_audio": x[-1], "input_video": x[1], "targets": x[0]},
                                    batch_size=config.batch_size, num_workers=config.workers, shuffle=False)

    loaders["valid"] = val_loader

    runner = SupervisedRunner(input_key=["input_audio", "input_video"]) # parameters of the model in forward(...)
    runner.infer(model, loaders,
                 callbacks=collections.OrderedDict({"snr_callback": SNRCallback(), "sdr_callback": SDRCallback()}),
                 verbose=True)

def main(args):
    config = ParamConfig(args.bs, args.epochs, args.workers, args.cuda, args.use_half, args.learning_rate)
    dataset = AVSpeechDataset(args.input_df_path, args.input_audio_size)
    val_dataset = AVSpeechDataset(args.val_input_df_path, args.input_audio_size)

    model, optimizer = load_best_model(conf)

    print(f"AVFusion has {sum(np.prod(i.shape) for i in model.parameters())}")

    validate(model, dataset, val_dataset, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--model-path", default="last_full.pth", type=str, help="trained model path")

    args = parser.parse_args()

    main(args)

    """
