#!/usr/bin/env bash

set -e -o pipefail

# This script is called from scripts like local/nnet3/run_tdnn.sh and
# local/chain/run_tdnn.sh (and may eventually be called by more scripts).  It
# contains the common feature preparation and iVector-related parts of the
# script.  See those scripts for examples of usage.


stage=0
nj=80
train_set=train_cleanup   # you might set this to e.g. train.
affix="_cleanup"
gmm=tri6b_cleanup # This specifies a GMM-dir from the features of the type you're training the system on;
                         # it should contain alignments for 'train_set'.
lang=data/lang_large_test
num_threads_ubm=32
num_processes=4
nnet3_affix="_cleanup" # affix for exp/nnet3 directory to put iVector stuff
ali_dir=exp/${gmm}_sp_ali
. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

for f in data/${train_set}/feats.scp $gmm/final.mdl; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done

if [ $stage -le 1 ]; then
  mfccdir=mfcc_sp
  #Although the nnet will be trained by high resolution data, we still have to
  # perturb the normal data to get the alignment.  _sp stands for speed-perturbed
  echo "$0: preparing directory for low-resolution speed-perturbed data (for alignment)"
  utils/data/perturb_data_dir_speed_3way.sh data/${train_set} data/${train_set}_sp
  echo "$0: making MFCC features for low-resolution speed-perturbed data"
  steps/make_mfcc_pitch_online.sh --nj $nj --cmd "$train_cmd" \
    data/${train_set}_sp exp/make_sp/${train_set}_sp $mfccdir || exit 1
  steps/compute_cmvn_stats.sh data/${train_set}_sp exp/make_sp/${train_set}_sp $mfccdir
  echo "$0: fixing input data-dir to remove nonexistent features, in case some "
  echo ".. speed-perturbed segments were too short."
  utils/fix_data_dir.sh data/${train_set}_sp
fi

if [ $stage -le 2 ]; then
  if [ -f $ali_dir/ali.1.gz ]; then
    echo "$0: alignments in $ali_dir appear to already exist.  Please either remove them "
    echo " ... or use a later --stage option."
    exit 1
  fi
  echo "$0: aligning with the perturbed low-resolution data"
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    data/${train_set}_sp $lang $gmm $ali_dir || exit 1
fi

if [ $stage -le 3 ]; then
  echo "$0: creating high-resolution MFCC features"

  # this shows how you can split across multiple file-systems.  we'll split the
  # MFCC dir across multiple locations.  You might want to be careful here, if you
  # have multiple copies of Kaldi checked out and run the same recipe, not to let
  # them overwrite each other.
  mfccdir=mfcc_hires_sp
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    utils/create_split_dir.pl /export/b0{5,6,7,8}/$USER/kaldi-data/mfcc/mandarin-$(date +'%m_%d_%H_%M')/s5/$mfccdir/storage $mfccdir/storage
  fi

  for datadir in ${train_set}_sp dev eval; do
    utils/copy_data_dir.sh data/$datadir data/${datadir}_hires
  done

  # do volume-perturbation on the training data prior to extracting hires
  # features; this helps make trained nnets more invariant to test data volume.
  utils/data/perturb_data_dir_volume.sh data/${train_set}_sp_hires

  for datadir in ${train_set}_sp_hires dev_hires eval_hires; do
    steps/make_mfcc_pitch_online.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/${datadir} exp/make_hires_sp/$datadir $mfccdir || exit 1
    steps/compute_cmvn_stats.sh data/${datadir} exp/make_hires_sp/$datadir $mfccdir
    utils/fix_data_dir.sh data/${datadir}
  
    # make MFCC data dir without pitch to extract iVector
    utils/data/limit_feature_dim.sh 0:39 data/${datadir} data/${datadir}_nopitch || exit 1;
    steps/compute_cmvn_stats.sh data/${datadir}_nopitch exp/make_hires_sp/${datadir}_nopitch $mfccdir || exit 1;
  done
fi

train_set=${train_set}_sp_hires_nopitch
if [ $stage -le 3 ]; then
  echo "Stage 3: train_set is $train_set"
  echo "$0: computing a subset of data to train the diagonal UBM."
  mkdir -p exp/nnet3${nnet3_affix}/diag_ubm
  temp_data_root=exp/nnet3${nnet3_affix}/diag_ubm

  # train a diagonal UBM using a subset of about a quarter of the data
  num_utts_total=$(wc -l <data/$train_set/utt2spk)
  num_utts=$[$num_utts_total/4]
  utils/data/subset_data_dir.sh data/$train_set\
      $num_utts ${temp_data_root}/${train_set}_subset

  echo "$0: computing a PCA transform from the hires data."
  steps/online/nnet2/get_pca_transform.sh --cmd "$train_cmd" \
      --splice-opts "--left-context=3 --right-context=3" \
      --max-utts 10000 --subsample 2 \
       ${temp_data_root}/${train_set}_subset \
       exp/nnet3${nnet3_affix}/pca_transform

  echo "$0: training the diagonal UBM."
  # Use 512 Gaussians in the UBM.
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj $nj \
    --num-frames 700000 \
    --num-threads $num_threads_ubm \
    ${temp_data_root}/${train_set}_subset 512 \
    exp/nnet3${nnet3_affix}/pca_transform exp/nnet3${nnet3_affix}/diag_ubm
fi

if [ $stage -le 4 ]; then
  # Train the iVector extractor.  Use all of the speed-perturbed data since iVector extractors
  # can be sensitive to the amount of data.  The script defaults to an iVector dimension of
  # 100.
  echo "$0: training the iVector extractor"
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 20 --num-processes $num_processes \
    data/${train_set} exp/nnet3${nnet3_affix}/diag_ubm exp/nnet3${nnet3_affix}/extractor || exit 1;
fi

if [ $stage -le 5 ]; then
  # note, we don't encode the 'max2' in the name of the ivectordir even though
  # that's the data we extract the ivectors from, as it's still going to be
  # valid for the non-'max2' data; the utterance list is the same.
  ivectordir=exp/nnet3${nnet3_affix}/ivectors_${train_set}
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $ivectordir/storage ]; then
    utils/create_split_dir.pl /export/b0{5,6,7,8}/$USER/kaldi-data/ivectors/mandarin-$(date +'%m_%d_%H_%M')/s5/$ivectordir/storage $ivectordir/storage
  fi
  # We extract iVectors on the speed-perturbed training data .  With
  # --utts-per-spk-max 2, the script pairs the utterances into twos, and treats
  # each of these pairs as one speaker; this gives more diversity in iVectors..
  # Note that these are extracted 'online' (they vary within the utterance).

  # Having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (the iVector starts at zero at the beginning
  # of each pseudo-speaker).
  temp_data_root=${ivectordir}
  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
    data/$train_set ${temp_data_root}/${train_set}_max2

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
    ${temp_data_root}/${train_set}_max2 \
    exp/nnet3${nnet3_affix}/extractor $ivectordir

  # Also extract iVectors for the test data, but in this case we don't need the speed
  # perturbation (sp).

  for data in dev eval; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
      data/${data}_hires_nopitch exp/nnet3${nnet3_affix}/extractor \
      exp/nnet3${nnet3_affix}/ivectors_${data}_hires_nopitch
  done
fi

exit 0;
