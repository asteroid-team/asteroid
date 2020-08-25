## Inspiration

#### DeMask's inspiration

We were having a fast call and one of us was in the train.
We couldn't hear him well because he was wearing a mask.
We directly thought about building surgically masked speech enhancement model with
Asteroid!
In the current covid pandemic situation, we can build better mask-adapted
speech technologies to help people keep their masks on and spread their words without
spreading the virus.

#### Asteroid's inspiration
It all started during a speech processing research project.
We wanted to go fast and tried "ready-to-use" speech separation/speech enhancement models.
We quickly realized that nothing was really working as expected,
and spent our time fixing other people's bugs instead of doing research.
This is the struggle that inspired Asteroid, and motivated us to open-source our
code with things that _just work_.

While sharing research code is already a step in the right direction,
sharing readable and reproducible code should be the standard.
Asteroid aims at empowering developers and researchers with tools that makes this even easier.

## What it does

#### About DeMask

DeMask is a simple, yet effective, end-to-end model to enhance speech when wearing face masks.
It restores the frequency content which is distorted by the face mask,
making the muffled speech sound cleaner.
The recipe to train the model is [here](https://github.com/mpariente/asteroid/tree/master/egs/demask) and the
pretrained model [here](https://zenodo.org/record/3997047#.X0Qw5Bl8Jkg),

#### About Asteroid

Asteroid is an audio source separation toolkit built with PyTorch and PyTorch-Lightning.
Inspired by the most successful neural source separation systems, it provides all neural
building  blocks  required  to  build  such  a  system.   To  improve  reproducibility,
recipes  on  common  audio source  separation datasets  are provided,
including all the steps from data download/preparation through training to evaluation.

Asteroid exposes all levels of granularity to the user from simple layers to complete ready-to-use models.
Our pretrained models are hosted on the [asteroid-models community in Zenodo](https://zenodo.org/communities/asteroid-models).
Loading pretrained models is trivial and sharing them is also made easy with asteroid's CLI.

You can check, [our landing page](https://asteroid-team.github.io/),
[our repo](https://github.com/mpariente/asteroid),
[our latest docs](https://mpariente.github.io/asteroid/)
and [our model hub](https://zenodo.org/communities/asteroid-models).

To try Asteroid, install the latest release with `pip install asteroid` or the
 current version with `pip install git+https://github.com/mpariente/asteroid` !


## How we built it

- Demask is trained on synthetic data generated from LibriSpeech's ([Panayotov et al. 2015](https://ieeexplore.ieee.org/document/7178964))
clean speech, distorted by approximate surgical mask finite impulse response (FIR) filters taken from [Corey et al. 2020](https://arxiv.org/abs/2008.04521).
The synthetic data is then augmented using room impulse responses (RIRs) from the FUSS dataset
([Wisdom et al. 2020](https://zenodo.org/record/3743844#.X0JZEhl8Jkg)).
A simple neural network estimates a time-frequency mask to correct the speech distortions.
Thanks to Asteroid's filterbanks (formulated using `torch.nn`), we could use a time domain loss with a time-frequency model
which yielded better results.


- Asteroid uses native PyTorch for layers and modules, a thin wrapper around
PyTorchLightning for training.
Most objects (models, filterbanks, optimizers, activation functions, normalizations) are retrievable
from string identifiers to improve efficiency on the command line.
Recipes are written in `bash` to separate data preparation, training and evaluation,
as adopted in Kaldi and ESPNet ASR toolkits.
During training, PyTorchLightning's coolest features (mixed-precision, distributed training, `torch_xla` support,
profiling, and more!) stay at the fingertips of our users.
We use Zenodo's REST API to upload and download pretrained models.

Our favorite `torch` op? We love `unfold` and `fold` and built cool end-to-end
signal processing tools with it !

## Examples

- Load DeMask's pretrained model using Hub or Asteroid

```python
# Without installing Asteroid
from torch import hub
model = hub.load(
"mpariente/asteroid", "demask", "popcornell/DeMask_Surgical_mask_speech_enhancement_v1"
)

# With asteroid install from master
from asteroid import DeMask
model = DeMask.from_pretrained("popcornell/DeMask_Surgical_mask_speech_enhancement_v1")
```

- Directly use it to enhance your muffled speech recordings using the `asteroid-infer` CLI

```bash
asteroid-infer popcornell/DeMask_Surgical_mask_speech_enhancement_v1 --files muffled.wav
```

To find the name of the model, we browsed our [Zenodo community](https://zenodo.org/communities/asteroid-models)
and picked the DeMask pretrained model
([snapshot here](https://challengepost-s3-challengepost.netdna-ssl.com/photos/production/software_photos/001/198/924/datas/original.png)).

- Train a speech separation model in 20 lines.

```python
from torch import optim
from pytorch_lightning import Trainer

from asteroid import ConvTasNet
from asteroid.losses import PITLossWrapper
from asteroid.data import LibriMix
from asteroid.engine import System

train_loader, val_loader = LibriMix.loaders_from_mini(task='sep_clean', batch_size=4)
model = ConvTasNet(n_src=2)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss = PITLossWrapper(
    lambda x, y: (x - y).pow(2).mean(-1),  # MSE
    pit_from="pw_pt",  # Point in the pairwise matrix.
)

system = System(model, optimizer, loss, train_loader, val_loader)
trainer = Trainer(fast_dev_run=True)
trainer.fit(system)
```

## Challenges we ran into

- In our first approach, we wanted to use the MASC dataset from the Compare challenge
(classification with unpaired data, [see here](http://www.compare.openaudio.eu/data/),
and use style transfer to perform enhancement but the amount of data was too small and the
differences between mask vs. no-mask too subtle. We suspect that surgical masks don't affect
speech as much as self-made coths masks do.

- When we got the impulse responses (IRs) from ([Corey et al. 2020](https://arxiv.org/abs/2008.04521)),
none of our first ideas worked because the filters contained the IR of the microphone and the room, and the phase
was noisy. We then resorted to design ad-hoc FIR filters which directly
approximate the frequency response of the masks in ([Corey et al. 2020](https://arxiv.org/abs/2008.04521))).
The filters were dynamically generated at training time to augment the available data.
Approximating the filters by hand saved us in the end !

- We would have loved to create a live demo of DeMask on the browser but the model
was not jitable, we'll definitely work on it in the future.


## Accomplishments that we are proud of
- Our `PITLossWrapper` takes any loss function and turns it into an efficient permutation invariant one !
Check out our [notebook about it](https://colab.research.google.com/github/mpariente/asteroid/blob/master/notebooks/03_PITLossWrapper.ipynb).
- Using Zenodo's REST API to automatize model sharing from the command line was
quite challenging and we believe it's a game changer to allow users to share their pretrained models.
- Giving proper credit is underrated: we're proud to release pretrained models with automatically-generated
appropriate license notices on them !
- We received the impulse responses from ([Corey et al. 2020](https://arxiv.org/abs/2008.04521))
less than a week before the challenge's deadline, ran into technical issues for generating
the training data but didn't quit !
- We've adapted PyTorch's sphinx template to create [our beautiful docs](https://mpariente.github.io/asteroid/).
- We've made our very own [landing page](https://asteroid-team.github.io/) and we love it.
- We've gathered [more than 20 contributors](https://github.com/mpariente/asteroid/graphs/contributors) from both academia and industry.
- We opened a [leaderboard in PapersWithCode](https://paperswithcode.com/sota/speech-separation-on-wsj0-2mix)
and we see new entries all the time.

## What we learned
Individually, we've learned to work as a team, set our goals, separate tasks and act fast.

## What's next

For DeMask, integrating end-to-end denoising and dereverberation with demasking
would make a good candidate for an open source version of NVIDIA RTX Voice.

For Asteroid, pretty much everything is next:
- A tighter integration with `torchaudio` and torch's `ComplexTensor`.
- `TorchScript` support.
- End to end separation to ASR with [ESPNet](https://github.com/espnet/espnet)
- Multi-channel extensions
- A nice refactoring of the bash recipes into Python CLI.
- A growing community of users and contributors.
and the list can go on..

## Acknowledgement

We'd like to thank all Asteroid's contributors !
<p align="center">
    •
    <a href="https://github.com/mhu-coder"> mhu-coder </a> •
    <a href="https://github.com/sunits"> sunits </a> •
    <a href="https://github.com/JorisCos"> JorisCos </a> •
    <a href="https://github.com/etzinis"> etzinis </a> •
    <a href="https://github.com/vitrioil"> vitrioil </a> •
    <a href="https://github.com/jensheit"> jensheit </a> •
    <a href="https://github.com/Ariel12321"> Ariel12321 </a> •
    <a href="https://github.com/tux-coder"> tux-coder </a> •
    <a href="https://github.com/saurabh-kataria"> saurabh-kataria </a> •
    <a href="https://github.com/subhanjansaha"> subhanjansaha </a> •
    <a href="https://github.com/mdjuamart"> mdjuamart </a> •
    <a href="https://github.com/hangtingchen"> hangtingchen </a> •
    <a href="https://github.com/groadabike"> groadabike </a> •
    <a href="https://github.com/dditter"> dditter </a> •
    <a href="https://github.com/bmorris3"> bmorris3 </a> •
    <a href="https://github.com/DizzyProtos"> DizzyProtos </a> •

</p>
