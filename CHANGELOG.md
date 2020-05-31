# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [Next release] - YYYY-MM-DD

#### Added 
#### Changed
#### Deprecated
#### Fixed

## [0.2.1] - 25/05/2020
#### Added 
- [src] Add dataset_name attribute to all data.Dataset (#113) (@mpariente)
- [hub] Add hubconf.py: load asteroid models without install ! (#112) (@mpariente)
- [src] Add support to the MUSDB18 dataset (#110) (@faroit)
- [src & tests] Importable models: ConvTasNet and DPRNNTasNet  (#109) (@mpariente)
#### Changed
- [src & tests] Depend on torch_optimizer for optimizers (#116) (@mpariente)
- [src & tests] Upgrade pytorch-lightning to >= 0.7.3 (#115) (@mpariente)
- [src] Reverts part of #112  (0.2.1 should be fully backward compatible) (@mpariente)
- [src] Change .pth convention for asteroid-models (#111) (@mpariente)
- [src] Split blocks in convolutional and recurrent (#107) (@mpariente)
- [install] Update pb_bss_eval to zero-mean si-sdr (@mpariente)
#### Deprecated
- [src] Deprecate kernel_size arg for conv_kernel_size in TDConvNet (#108) (@mpariente)
- [src] Deprecate masknn.blocks (splited) (#107) (@mpariente)
#### Fixed
- [src] Fix docstring after #108 (@mpariente)
- [egs] Replace PITLossWrapper arg mode by pit_from (#103) (@mpariente)


## [0.2.0] - 08/05/2020
#### Added 
- [egs] Deep clustering/Chimera++ recipe (#96) (@mpariente)
- [src & egs] Source changes towards deep clustering recipe (#95) (@mpariente)
- [docs] Add training logic figure (#94) (@mpariente)
- [install] Include PMSQE matrices in setup.py (@mpariente)
- [src & egs] DPRNN architecture change + replicated results (#93) (@mpariente)
- [egs] Two step recipe : update results (#91) (@etzinis)
- [src & tests] Add multi-phase gammatone filterbank (#89) (@dditter)
- [src] Stabilize consistency constraint (@mpariente)
- [src] LibriMix dataset importable from data (@mpariente)
- [src & egs] LibriMix dataset support and ConvTasnet recipe (#87) (@JorisCos)
- [egs] Dynamic mixing for Wham (#80) (@popcornell)
- [egs] Add FUSS data preparation files (#86) (@michelolzam)
- [src & tests] Implement MISI (#85) (@mpariente)
- [src] ConvTasnetv1 available : no skip option to TDCN (#82) (@mpariente)
- [src & tests] Implement GriffinLim (#83) (@mpariente)
- [docs] Add reduce example in PITLossWrapper (@mpariente)
- [src & tests] Generalize ! Implement pairwise losses reduce function (#81) (@mpariente)
- [src & tests] Add support for STOI loss (#79) (@mpariente)
- [src] Support padding and output_padding in Encoder and Decoder (#78) (@mpariente)
- [src & tests] Added mixture consistency constraints (#77) (@mpariente)
- [logo] Add white-themed asteroid logo (@mpariente)
- [egs] Add Two step source separation recipe (#67) (@etzinis)
- [src] Add train kwarg in System's common_step for easier subclassing. (@mpariente)
- [egs] Upload Tasnet WHAMR results (@mpariente)
- [src & tests] Add PMSQE loss in asteroid (#65) (@mdjuamart)
- [src] Add Ranger in supported optimizers (@mpariente)

#### Changed
- [egs] Remove python installs in recipes (#100) (@mpariente)
- [egs] Remove abs path in recipes (#99) (@mpariente)
- [egs] Add DC requirements.txt (@mpariente)
- [egs] Remove file headers (@mpariente)
- [src] Remove Chimerapp from blocks.py (#98) (@mpariente)
- [egs] Better logging : copy logs to expdir (#102) (@mpariente)
- [src] Delete wav.py (@mpariente)
- [egs] Delete unused file (@mpariente)
- [ci] Restore old travis.yml (reverts part of #61) (@mpariente)
- [egs] Utils symlink sms_wsj (@mpariente)
- [egs] Utils symlink wham/TwoStep (@mpariente)
- [egs] Utils symlink wham/DPRNN (@mpariente)
- [egs] Utils symlink LibriMix (@mpariente)
- [src & egs] Replace WSJ0-mix Dataset (#97) (@mpariente)

#### Fixed
- [src] Return config in DPRNN.get_config() (@mpariente)
- [egs] Fix typo in LibriMix import (@mpariente)
- [egs] Twostep recipe small fixes (#74) (@mpariente)
- [egs] Fixed mode overwriting in all wham recipes (#76) (@Ariel12321)
- [egs] Fixed mode overwriting in ConvTasNet recipe (#75) (@Ariel12321)
- Fix build erros (Pin sphinx < 3.0) (#72) (@mpariente)
- Fixing paths for wham scripts and unziping command for noise (#66) (@etzinis)
- Fix whamr_dataset.py, map reverb to anechoic (@mpariente)
- Important fix : WHAMR tasks include dereverberation ! (@mpariente)

