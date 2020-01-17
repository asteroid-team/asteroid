import sys
#to import module from directories
sys.path.extend(["models", "train", "loader"])

import torch
import numpy as np
from pathlib import Path
from trainer import train
from config import ParamConfig
from data_loader import AVDataset
from memory_profiler import profile
from argparse import ArgumentParser
from models import Audio_Visual_Fusion as AVFusion


class DiscriminativeLoss(torch.nn.Module):

    def __init__(self, gamma=0.1):
        super(DiscriminativeLoss, self).__init__()

        self.gamma = gamma
        self.mse_loss = torch.nn.MSELoss(reduction="mean")

    def forward(self, input, target):
        swapped_target = target.clone().detach()
        swapped_target[..., 0], swapped_target[..., 1] = target[..., 1], target[..., 0]

        loss = self.mse_loss(input, target)
        loss -= self.gamma * self.mse_loss(input, swapped_target)

        return loss


def main(args):
    config = ParamConfig(args.bs, args.epochs, args.workers, args.cuda, args.use_half)
    dataset = AVDataset(args.dataset_path, args.video_dir,
                        args.input_df_path, args.input_audio_size, args.cuda)
    val_dataset = AVDataset(args.dataset_path, args.video_dir,
                        args.val_input_df_path, args.input_audio_size, args.cuda)

    if args.cuda:
        device = torch.device("cuda:0")
        model = AVFusion(num_person=args.input_audio_size, device=device).train()
        model = model.to(device)
    else:
        model = AVFusion(num_person=args.input_audio_size).train()


    p = "logdir/checkpoints/last_full.pth"
    if Path(p).is_file():
        ckpt = torch.load(p)
        print(list(ckpt.keys()))
    print(f"AVFusion has {sum(np.prod(i.shape) for i in model.parameters())}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.MSELoss(reduction="mean")#DiscriminativeLoss()#

    train(model, dataset, optimizer, criterion, config, val_dataset=val_dataset, validate=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--bs", default=16, type=int, help="batch size of dataset")
    parser.add_argument("--epochs", default=40, type=int, help="max epochs to train")
    parser.add_argument("--cuda", default=True, type=bool, help="cuda for training")
    parser.add_argument("--workers", default=0, type=int, help="total workers for dataset")
    parser.add_argument("--input-audio-size", default=2, type=int, help="total input size")
    parser.add_argument("--dataset-path", default=Path("../data/audio_visual/avspeech_train.csv"), type=Path, help="path for avspeech training data")
    parser.add_argument("--video-dir", default=Path("../data/train"), type=Path, help="directory where all videos are stored")
    parser.add_argument("--input-df-path", default=Path("more_train.csv"), type=Path, help="path for combinations dataset")
    parser.add_argument("--val-input-df-path", default=Path("more_val.csv"), type=Path, help="path for combinations dataset")
    parser.add_argument("--use-half", default=False, type=bool, help="halves the precision")
    parser.add_argument("--learning-rate", default=3e-4, type=float, help="learning rate for the network")

    args = parser.parse_args()

    main(args)
