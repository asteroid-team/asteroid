import sys

import collections
from pathlib import Path
from argparse import ArgumentParser

import torch
import numpy as np
from catalyst.dl import utils
from catalyst.dl.runner import SupervisedRunner

from src.loader import AVDataset
from src.models import Audio_Visual_Fusion as AVFusion
from src.train import SNRCallback, SDRCallback, ParamConfig

class DiscriminativeLoss(torch.nn.Module):

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
    dataset = AVDataset(args.input_df_path, args.input_audio_size)
    val_dataset = AVDataset(args.val_input_df_path, args.input_audio_size)

    if args.cuda:
        device = torch.device("cuda:0")
        model = AVFusion(num_person=args.input_audio_size, device=device).train()
        model = model.to(device)
    else:
        model = AVFusion(num_person=args.input_audio_size).eval()

    if Path(args.model_path).is_file():
        ckpt = torch.load(args.model_path)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print(f"Model doesn't exist at {args.model_path}")
        return
    print(f"AVFusion has {sum(np.prod(i.shape) for i in model.parameters())}")

    validate(model, dataset, val_dataset, config)


if __name__ == "__main__":
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

