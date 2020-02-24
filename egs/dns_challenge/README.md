### Adbout the DNS Challenge

The Deep Noise Suppression (DNS) Challenge is a single-channel speech enhancement 
challenge organized by Microsoft, with a focus on real-time applications. 
More info can be found on the [official page](https://dns-challenge.azurewebsites.net/).


### - Our basic recipe
This is made to make *your* life simpler and research easier !
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
- All the research, all the fun !

##### How to use it? 
- Need to install a python environment? 
[Check this out]() !
- Open `run.sh`, change `storage_dir` to a path where you can afford storing 
320GB of data.
- Just `./run.sh` and it's on. 

#####Some notes:
- After the first execution, you can go and change `stage=4` in `run.sh` to 
avoid redoing all the steps everytime. 
- To use GPUs for training, run `run.sh --id 0,1` where `0` and `1` are the 
GPUs you want to use, training will automatically take advantage of both GPUs.
- By default, a random id is generated for each run, you can also add a 
`tag` to name the experiments how you want. For example 
`run.sh --tag with_cool_loss` will save all results to 
`exp/train_dns_with_cool_loss`. You'll also find the corresponding log 
file in `logs/train_dns_with_cool_loss.log`.


The data download, dataset creation and preprocessing will take a while 
(around a day in my case). From stage 4 (training), be sure you have 
enough compute power to train your DNN. Before that, you're I/O bound so 
not much compute power is needed.

### References
- The challenge paper, [here](https://arxiv.org/abs/2001.08662).
```BibTex
@misc{DNSChallenge2020,
title={The INTERSPEECH 2020 Deep Noise Suppression Challenge: Datasets, Subjective Speech Quality and Testing Framework},
author={Chandan K. A. Reddy and Ebrahim Beyrami and Harishchandra Dubey and Vishak Gopal and Roger Cheng and Ross Cutler and Sergiy Matusevych and Robert Aichner and Ashkan Aazami and Sebastian Braun and Puneet Rana and Sriram Srinivasan and Johannes Gehrke}, year={2020},
eprint={2001.08662},
}
```
- The baseline paper, [here](https://arxiv.org/abs/2001.10601).
```BibTex
@misc{xia2020weighted,
title={Weighted Speech Distortion Losses for Neural-network-based Real-time Speech Enhancement},
author={Yangyang Xia and Sebastian Braun and Chandan K. A. Reddy and Harishchandra Dubey and Ross Cutler and Ivan Tashev},
year={2020},
eprint={2001.10601},
}
```