#!/bin/bash


sphere_dir= # Directory containing sphere files
wav_dir= # Directory where to save all wav files (Need disk space)
data_dir=data


stage=0
tag=""

. utils/parse_options.sh || exit 1;

EXP_ROOT=$PWD

if [ $stage -le  0 ]; then
  . local/convert_sphere2wav.sh --sphere_dir $sphere_dir --wav_dir $wav_dir
fi


if [ $stage -le  1 ]; then
  . local/prepare_data.sh --wav_dir $wav_dir
fi

if [ $stage -le  2 ]; then
  python preprocess.py --wav_dir $wav_dir --data_dir
fi





#if [ -z ${tag} ]; then
#  tag1=r${sample_rate}_noisy${noisy}_${mode}_${filt_type}_${inp_type}_mask${mask_type}_encdec${enc_dec}_sincos${sine_cos}
#  tag2=${tag1}_co${cos_only}_N${N}_L${L}_B${B}H${H}P${P}X${X}R${R}C${C}${norm_type}causal${causal}${mask_nonlinear}
#  tag3=${tag2}_epoch${epochs}_half${half_lr}norm${max_norm}bs${batch_size}_${optimizer}
#  tag=${tag3}_lr${lr}_mmt${momentum}_l2${l2}
#fi
#
#expdir=exp/train_${tag}
#echo $expdir
#mkdir -p logs


#if [ $stage -le 3 ]; then
#  python train.py
#fi
#
#if [ $stage -le 4 ]; then
#  python visualize.py
#fi
#
#if [ $stage -le 5 ]; then
#  python evaluate.py
#fi


#
#
#data_dir=
#wsj0_base=
#
## WHAM noises
#wget https://storage.googleapis.com/whisper-public/wham_noise.zip $data_dir
#unzip
#mv wham_noise.zip
#
## WHAM Scripts and docs
#wget https://storage.googleapis.com/whisper-public/wham_scripts.tar.gz $data_dir
#run scripts
## Preprocessing (do all the cases)
## Find out wsj-3mix as well no?
#
#
## Training
#
#
## Evaluation
#
#
## Visualization
#
#
#
## WSJ0 base folder
#wrami=g5k # Where am I ?
#dataset=2speakers_wham
#
#stage=0  # Modify this to control to start from which stage
#
#id=$CUDA_VISIBLE_DEVICES #GPU ID
## Dataset config
#sample_rate=8000
#noisy=0
#mode=min
#small=0
#segment=4  # seconds
#max_len=16 # seconds, used only if batch_size=1
#cv_maxlen=6  # seconds
#
## Network config
#N=512
#L=16
#B=128
#H=512
#P=3
#X=6 #8
#R=2 #3
#norm_type=gLN
#causal=0
#mask_nonlinear='relu'
#C=2
## Training config
#use_cuda=1
#epochs=400 #100
#half_lr=1
#early_stop=1
#max_norm=5
## minibatch
#batch_size=3
#num_workers=6
## optimizer
#optimizer=adam
#load_optimizer=1
#lr=1e-3
#momentum=0
#l2=0
## save and visualize
#checkpoint=0
#continue_from=""
#model_path=final.pth.tar
#norm_type=gLN
#use_cuda=1
#print_freq=1000
#
## Encoder decoder config
#filt_type=learned
#inp_type=cat
#mask_type=separate #same
#enc_dec=0
#sine_cos=1
#cos_only=0
#stft_use_N=0
#
## evaluate
#eval_dir=""
#ev_use_cuda=0
#cal_sdr=1
#cal_pesq=0
#cal_stoi=0
#test_subset=0 #0 = all
#save_wavs=50 # Number of wavs to save
#
#tag=""
#
#. utils/parse_options.sh || exit 1;
#
#if [ $wrami = g5k ]; then
#  wsj0_base=/talc3/multispeech/calcul/users/mpariente/DATA/wsj0_wav
#elif [ $wrami = xplor ]; then
#  wsj0_base=/mnt/beegfs/pul51/zaf67/wsj0_wav
#elif [ $wrami = local ]; then
#  wsj0_base=data/wsj0_wav
#fi
#
## Get appropriate folder
#sr_string=$(($sample_rate/1000))
#suffix=$dataset/wav${sr_string}k/$mode
#data=$wsj0_base/$suffix
#
#if [ $small = 0 ]; then
#  dumpdir=data/$suffix  # directory to put generated json file
#else
#  dumpdir=data_small/$suffix  # directory to put generated json file
#  if [ -z ${tag} ]; then
#    tag=tmp
#  fi
#fi
#train_dir=$dumpdir/tr
#valid_dir=$dumpdir/cv
#evaluate_dir=$dumpdir/tt
#
#
#if [ $stage -le 0 ]; then
#  echo "Stage 0: Generating json files including wav path and duration"
#  [ ! -d $dumpdir ] && mkdir -p $dumpdir
#  python preprocess.py --in-dir $data --out-dir $dumpdir --noisy $noisy --small_test $small --sample_rate $sample_rate
#fi
#
#if [ -z ${tag} ]; then
#  tag1=r${sample_rate}_noisy${noisy}_${mode}_${filt_type}_${inp_type}_mask${mask_type}_encdec${enc_dec}_sincos${sine_cos}
#  tag2=${tag1}_co${cos_only}_N${N}_L${L}_B${B}H${H}P${P}X${X}R${R}C${C}${norm_type}causal${causal}${mask_nonlinear}
#  tag3=${tag2}_epoch${epochs}_half${half_lr}norm${max_norm}bs${batch_size}_${optimizer}
#  tag=${tag3}_lr${lr}_mmt${momentum}_l2${l2}
#fi
#
#expdir=exp/train_${tag}
#echo $expdir
#mkdir -p logs
#
#if [ $stage -le 1 ]; then
#  echo "Stage 1: Training"
#  mkdir -p $expdir
#  CUDA_VISIBLE_DEVICES=$id python train.py \
#  --train_dir $train_dir \
#  --valid_dir $valid_dir \
#  --noisy $noisy \
#  --use_cuda $use_cuda \
#  --sample_rate $sample_rate \
#  --segment $segment \
#  --cv_maxlen $cv_maxlen \
#  --max_len $max_len \
#  --N $N \
#  --L $L \
#  --B $B \
#  --H $H \
#  --P $P \
#  --X $X \
#  --R $R \
#  --C $C \
#  --norm_type $norm_type \
#  --causal $causal \
#  --mask_nonlinear $mask_nonlinear \
#  --use_cuda $use_cuda \
#  --epochs $epochs \
#  --half_lr $half_lr \
#  --early_stop $early_stop \
#  --max_norm $max_norm \
#  --batch_size $batch_size \
#  --num_workers $num_workers \
#  --optimizer $optimizer \
#  --load_optimizer $load_optimizer \
#  --lr $lr \
#  --momentum $momentum \
#  --l2 $l2 \
#  --save_folder ${expdir} \
#  --checkpoint $checkpoint \
#  --continue_from "$continue_from" \
#  --model_path $model_path \
#  --filt_type $filt_type \
#  --inp_type $inp_type \
#  --mask_type $mask_type \
#  --enc_dec $enc_dec \
#  --sine_cos $sine_cos \
#  --cos_only $cos_only \
#  --stft_use_N $stft_use_N \
#  --print_freq $print_freq \
#  --small_test $small | tee logs/train_${tag}.log
#fi
#
#if [ -z ${eval_dir} ]; then
#  eval_dir=$expdir
#fi
#if [ $stage -le 2 ]; then
#  echo "Stage 2: Evaluation"
#  # Model path overrides data_dir, noisy and
#  # sample_rate --> Start from stage 2 = ok.
#  CUDA_VISIBLE_DEVICES=$id python evaluate.py \
#  --model_path ${eval_dir}/$model_path \
#  --data_dir $data/tt/ \
#  --noisy $noisy \
#  --cal_sdr $cal_sdr \
#  --cal_pesq $cal_pesq \
#  --cal_stoi $cal_stoi \
#  --use_cuda $ev_use_cuda \
#  --sample_rate $sample_rate \
#  --test_subset $test_subset \
#  --save_wavs $save_wavs
#fi
#
#if [ $stage -le 3 ]; then
#  echo "Stage 3 : Visualize"
#  python visualize.py \
#  --exp_dir ${eval_dir} \
#  --model_path $model_path
#fi
#
#
