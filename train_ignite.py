import os
import sys
import time
# to import module from directories
sys.path.extend(["models", "loader", "utils"])

import torch
import numpy as np
from pathlib import Path
# from train/
#from trainer import train
#from config import ParamConfig
# from loader/
from data_loader import AVDataset
from memory_profiler import profile
from argparse import ArgumentParser
# from  models/
from models import Audio_Visual_Fusion as AVFusion

from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

from ignite.metrics import Loss
from ignite.handlers import ModelCheckpoint
from ignite.engine import (Engine, Events, create_supervised_trainer,
                           create_supervised_evaluator)
from ignite.contrib.handlers.param_scheduler import LRScheduler

# from utils/
from utils import SaveAudio
from config import ParamConfig
from loss_utils import DiscriminativeLoss
from metric_utils import snr, SNRMetric, SDRMetric

ID = time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime())
print(ID)
def main(args):
    
    device = "cpu"
    if args.cuda:
        device = "cuda"
    device = torch.device(device)

    # create training and validation dataset
    dataset = AVDataset(args.dataset_path, args.video_dir,
                        args.train_path, args.input_audio_size, args.cuda)
    val_dataset = AVDataset(args.dataset_path, args.video_dir,
                        args.val_path, args.input_audio_size, args.cuda)

    config = ParamConfig(args.bs, args.epochs, args.workers, args.cuda, args.use_half)

    # create model and assign device
    model = AVFusion(num_person=args.input_audio_size, device=device)
    model.to(device)

    if args.saved_model_path:
        print(f"loading {args.saved_model_path}")
        model.load_state_dict(torch.load(args.saved_model_path()))
        print("loaded")

    # set optimizer, lr, criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.MSELoss(reduction="mean")#DiscriminativeLoss()

    # create the dataloader
    train_loader = torch.utils.data.DataLoader(dataset, config.batch_size, shuffle=True,
                                                num_workers=config.workers)

    val_loader = torch.utils.data.DataLoader(val_dataset, config.batch_size, shuffle=True,
                                                num_workers=config.workers)

    # accumulation step for gradient accumulation
    accumulation_step = args.accumulation_step

    # learning rate scheduler
    scheduler = StepLR(optimizer, step_size=(1.8e6) // config.batch_size, gamma=0.5)
    scheduler = LRScheduler(scheduler)

    # define the metrics
    loss = Loss(criterion)
    snr_metric = SNRMetric()
    sdr_metric = SDRMetric()

    # custom update function for train to handle the batch
    def _update(engine, batch):
        #model.train()

        target, input_video, input_audio = batch

        target = target.to(device)
        input_audio = input_audio.to(device)
        input_video = [i.to(device) for i in input_video]

        output = model(input_audio, input_video)

        # distribute the loss
        loss = criterion(output, target) / accumulation_step
        loss.backward()

        # accumulate gradients and update at fixed point of time
        if engine.state.iteration % accumulation_step == 0:
            optimizer.zero_grad()
            optimizer.step()

        snr_value = snr(output, target)

        return {"dis_loss": loss.item(), "y_pred": output,
                "y": target, "snr": snr_value}

    # custom inference function
    def _inference(engine, batch):
        #model.eval()

        with torch.no_grad():
            target, input_video, input_audio = batch

            target = target.to(device)
            input_audio = input_audio.to(device)
            input_video = [i.to(device) for i in input_video]

            output = model(input_audio, input_video)

        return output, target

    # set the trainer and evaluator
    trainer = Engine(_update)
    evaluator = Engine(_inference)

    # attach the metric hooks to evaluator
    loss.attach(evaluator, "dis_loss")
    snr_metric.attach(evaluator, "snr")
    #sdr_metric.attach(evaluator, "sdr")  

    # define the logger
    loss_desc = "EPOCH: {}/{} ITERATION: {}/{} - Loss: {:.4f} SNR: {:.4f}"
    loss_pbar = tqdm(initial=0, leave=True, total=len(train_loader),
                     desc=loss_desc.format(0, 0, 0, 0, 0, 0))
    log_interval = accumulation_step

    iter_desc = "ITERATION: {}/{}"
    iter_pbar = tqdm(initial=0, leave=False, total=len(val_loader),
                     desc=iter_desc.format(0, 0))

    # print loss for trainer
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        loss_pbar.set_description(loss_desc.format(engine.state.epoch, config.epochs,
                                  engine.state.iteration % len(train_loader),
                                  len(train_loader), engine.state.output["dis_loss"],
                                  engine.state.output["snr"]))
        loss_pbar.update(1)

    # print validation informarion after epoch completion
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        model.eval()

        iter_pbar.refresh()
        loss_pbar.refresh()

        iter_pbar.reset()
        loss_pbar.reset()

        # evaluate the validation data
        evaluator.run(val_loader)
        # retrieve the metrics and print
        metrics = evaluator.state.metrics

        avg_snr = metrics.get("snr", np.nan)
        avg_dis_loss = metrics.get("dis_loss", np.nan)
        avg_sdr = metrics.get("sdr", np.nan)
        
        log_res = f"Validation Results - Epoch: {engine.state.epoch} Avg loss: {avg_dis_loss:.2f} AVG snr: {avg_snr:.4f} AVG sdr: {avg_sdr:.4f}"
        print(f"\n\n{log_res}\n\n")
        epoch = engine.state.epoch
        with open(f"logs/val_log_{ID}_EPOCH_{epoch}.txt", 'a') as f:
            tqdm.write(log_res, f)
        model.train()

    # iterator while predicting validation data
    @evaluator.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter_pbar.set_description(iter_desc.format(engine.state.iteration % len(val_loader), len(val_loader)))
        iter_pbar.update(1)

    os.mkdir(f"models_{ID}")
    checkpoint = ModelCheckpoint(dirname=f"models_{ID}/", filename_prefix=f"model_{ID}_",
                                 save_interval=1, n_saved=2, create_dir=True)
    save_audio = SaveAudio(dirname="output/", filename_prefix="audio_")

    # set the event handlers
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler) # set the lr scheduler
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint, {"model": model}) # set the checkpointer
    #lots of writing slows down
    #trainer.add_event_handler(Events.ITERATION_COMPLETED, save_audio) # save audio

    # run the trainer
    trainer.run(train_loader, max_epochs=config.epochs)

    loss_pbar.close()
    iter_pbar.close()


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--bs", default=16, type=int, help="batch size of dataset")
    parser.add_argument("--epochs", default=10, type=int, help="max epochs to train")
    parser.add_argument("--cuda", default=True, type=bool, help="cuda for training")
    parser.add_argument("--workers", default=0, type=int, help="total workers for dataset")
    parser.add_argument("--input-audio-size", default=2, type=int, help="total input size")
    parser.add_argument("--dataset-path", default=Path("../data/audio_visual/avspeech_train.csv"), type=Path, help="path for avspeech training data")
    parser.add_argument("--video-dir", default=Path("../data/train"), type=Path, help="directory where all videos are stored")
    parser.add_argument("--train-path", default=Path("train.csv"), type=Path, help="path for combinations dataset")
    parser.add_argument("--val-path", default=Path("val.csv"), type=Path, help="path for val combinations dataset")
    parser.add_argument("--use-half", default=False, type=bool, help="halves the precision")
    parser.add_argument("--learning-rate", default=3e-5, type=float, help="learning rate for the network")
    parser.add_argument("--accumulation-step", default=1, type=int, help="accumulation steps")
    parser.add_argument("--saved-model-path", default=None, type=str, help="saved model path")

    args = parser.parse_args()

    main(args)

