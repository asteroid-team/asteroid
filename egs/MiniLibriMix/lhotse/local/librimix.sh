#!/usr/bin/env bash

# modified from https://github.com/pzelasko/lhotse/tree/master/examples/librimix author: Piotr Żelasko

set -eou pipefail

LIBRIMIX_ROOT=$1
OUT_PATH=$2
DURATION=$3
LIBRIMIX_CSV=${LIBRIMIX_ROOT}/MiniLibriMix/metadata/mixture_train_mix_both.csv


[[ `uname` == 'Darwin' ]] && nj=`sysctl -n machdep.cpu.thread_count` || nj=`grep -c ^processor /proc/cpuinfo`


# Prepare audio and supervision manifests
lhotse recipe librimix \
  --min-segment-seconds $DURATION \
  --with-precomputed-mixtures \
  ${LIBRIMIX_CSV} \
  $OUT_PATH

for type in sources mix noise; do
  # Extract features for each type of audio file
  lhotse make-feats -j $nj \
    -r ${LIBRIMIX_ROOT} \
    $OUT_PATH/audio_${type}.yml \
    $OUT_PATH/feats_${type}
  # Create cuts out of features - cuts_mix.yml will contain pre-mixed cuts for source separation
  lhotse cut simple \
    -s $OUT_PATH/supervisions_${type}.yml \
    $OUT_PATH/feats_${type}/feature_manifest.yml.gz \
    $OUT_PATH/cuts_${type}.yml.gz
done

# Prepare cuts with feature-domain mixes performed on-the-fly - clean
lhotse cut mix-by-recording-id $OUT_PATH/cuts_sources.yml.gz $OUT_PATH/cuts_mix_dynamic_clean.yml.gz
# Prepare cuts with feature-domain mixes performed on-the-fly - noisy
lhotse cut mix-by-recording-id $OUT_PATH/cuts_sources.yml.gz $OUT_PATH/cuts_noise.yml.gz $OUT_PATH/cuts_mix_dynamic_noisy.yml.gz

# The next step is truncation - it makes sure that the cuts all have the same duration and makes them easily batchable

# Truncate the pre-mixed cuts
lhotse cut truncate \
  --max-duration $DURATION \
  --offset-type random \
  --preserve-id \
  $OUT_PATH/cuts_mix.yml.gz $OUT_PATH/cuts_mix_${DURATION}s.yml.gz

# Truncate the dynamically-mixed clean cuts
lhotse cut truncate \
  --max-duration $DURATION \
  --offset-type random \
  --preserve-id \
  $OUT_PATH/cuts_mix_dynamic_clean.yml.gz $OUT_PATH/cuts_mix_dynamic_clean_${DURATION}s.yml.gz

# Truncate the dynamically-noisy clean cuts
lhotse cut truncate \
  --max-duration $DURATION \
  --offset-type random \
  --preserve-id \
  $OUT_PATH/cuts_mix_dynamic_noisy.yml.gz $OUT_PATH/cuts_mix_dynamic_noisy_${DURATION}s.yml.gz

# Processing complete - the resulting YAML mixed cut manifests can be loaded in Python to create a PyTorch dataset.