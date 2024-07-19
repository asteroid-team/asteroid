This is made to make *your* life simpler and research easier!
##### What we automate for you :
- Install `git-lfs` without root (required to download the data).
- Download the data from the official repo.
- Create the dataset with default parameters.
- Ready-to-use `DataLoader` to train your net with.
- Example scripts with all the ingredients for a successful system
- MutliGPU support / Logging (+ Tensorboard) / LR scheduler (thanks
[lightning](https://github.com/PyTorchLightning/pytorch-lightning)!)

##### What you can focus on :
- Some new architectures to outperform our model
- Fancy loss functions to improve speech quality
- All the research, all the fun!

##### How to use it?
- Need to install a python environment?
[Check this out]()!
- Open `run.sh`, change `storage_dir` to a path where you can afford storing
320GB of data.
- Just `./run.sh` and it's on.

##### Some notes:
- After the first execution, you can go and change `stage=4` in `run.sh` to
avoid redoing all the steps everytime.
- To use GPUs for training, run `run.sh --id 0,1` where `0` and `1` are the
GPUs you want to use, training will automatically take advantage of both GPUs.
- By default, a random id is generated for each run, you can also add a
`tag` to name the experiments how you want. For example
`run.sh --tag with_cool_loss` will save all results to
`exp/train_dns_with_cool_loss`. You'll also find the corresponding log
file in `logs/train_dns_with_cool_loss.log`.
- If you want to change the data generation config, go the `storage_dir`,
change the noisyspeech_synthesizer.cfg accordingly and restart from stage 2.
Be aware that this will overwrite the previous json files in `data/`.

The data download, dataset creation and preprocessing will take a while
(around a day in my case). From stage 4 (training), be sure you have
enough compute power to train your DNN. Before that, you're I/O bound so
not much compute power is needed.
