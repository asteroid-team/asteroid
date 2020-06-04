#!/usr/bin/env bash

# Copyright 2016 Pegah Ghahremani

# This script dumps bottleneck feature for model trained using nnet3.
# CAUTION!  This script isn't very suitable for dumping features from recurrent
# architectures such as LSTMs, because it doesn't support setting the chunk size
# and left and right context.  (Those would have to be passed into nnet3-compute).
# See also chain/get_phone_post.sh.

# Begin configuration section.
stage=1
nj=4
cmd=queue.pl
use_gpu=false
ivector_dir=
compress=true
# End configuration options.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [[ ( $# -lt 4 ) || ( $# -gt 6 ) ]]; then
   echo "usage: steps/nnet3/make_bottleneck_features.sh <bnf-node-name> <input-data-dir> <bnf-data-dir> <nnet-dir> [<log-dir> [<bnfdir>] ]"
   echo "e.g.:  steps/nnet3/make_bottleneck_features.sh tdnn_bn.renorm data/train data/train_bnf exp/nnet3/tdnn_bnf exp_bnf/dump_bnf bnf"
   echo "Note: <log-dir> defaults to <bnf-data-dir>/log and <bnfdir> defaults to"
   echo " <bnf-data-dir>/data"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --ivector-dir                                    # directory for ivectors"
   exit 1;
fi
bnf_name=$1 # the component-node name in nnet3 model used for bottleneck feature extraction
data=$2
bnf_data=$3
nnetdir=$4
if [ $# -gt 4 ]; then
  logdir=$5
else
  logdir=$bnf_data/log
fi
if [ $# -gt 5 ]; then
  bnfdir=$6
else
  bnfdir=$bnf_data/data
fi

# Assume that final.nnet is in nnetdir
cmvn_opts=`cat $nnetdir/cmvn_opts`;
bnf_nnet=$nnetdir/final.raw
if [ ! -f $bnf_nnet ] ; then
  if [ ! -f $nnetdir/final.mdl ]; then
    echo "$0: No such file $bnf_nnet or $nnetdir/final.mdl";
    exit 1;
  else
    bnf_nnet=$nnetdir/final.mdl
  fi
fi

if $use_gpu; then
  compute_queue_opt="--gpu 1"
  compute_gpu_opt="--use-gpu=yes"
  if ! cuda-compiled; then
    echo "$0: WARNING: you are running with one thread but you have not compiled"
    echo "   for CUDA.  You may be running a setup optimized for GPUs.  If you have"
    echo "   GPUs and have nvcc installed, go to src/ and do ./configure; make"
    exit 1
  fi
else
  echo "$0: without using a GPU this will be very slow.  nnet3 does not yet support multiple threads."
  compute_gpu_opt="--use-gpu=no"
fi


## Set up input features of nnet
name=`basename $data`
sdata=$data/split$nj

mkdir -p $logdir
mkdir -p $bnf_data
mkdir -p $bnfdir
echo $nj > $bnfdir/num_jobs

[ ! -f $data/feats.scp ] && echo >&2 "The file $data/feats.scp does not exist!" && exit 1;
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

use_ivector=false
if [ ! -z "$ivector_dir" ];then
  use_ivector=true
  steps/nnet2/check_ivectors_compatible.sh $nnetdir $ivector_dir || exit 1;
fi

## Set up features.
if [ -f $nnetdir/online_cmvn ]; then online_cmvn=true
else online_cmvn=false; fi

if ! $online_cmvn; then
  echo "$0: feature type is raw"
  feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"
else
  echo "$0: feature type is raw (apply-cmvn-online)"
  feats="ark,s,cs:apply-cmvn-online $cmvn_opts --spk2utt=ark:$sdata/JOB/spk2utt $nnetdir/global_cmvn.stats scp:$sdata/JOB/feats.scp ark:- |"
fi
ivector_feats="scp:utils/filter_scp.pl $sdata/JOB/utt2spk $ivector_dir/ivector_online.scp |"

if [ $stage -le 1 ]; then
  echo "$0: Generating bottleneck (BNF) features using $bnf_nnet model as output of "
  echo "    component-node with name $bnf_name."
  echo "output-node name=output input=$bnf_name" > $bnf_data/output.config
  modified_bnf_nnet="nnet3-copy --nnet-config=$bnf_data/output.config $bnf_nnet - |"
  ivector_opts=
  if $use_ivector; then
    ivector_period=$(cat $ivector_dir/ivector_period) || exit 1;
    ivector_opts="--online-ivector-period=$ivector_period --online-ivectors='$ivector_feats'"
  fi
  $cmd $compute_queue_opt JOB=1:$nj $logdir/make_bnf_$name.JOB.log \
    nnet3-compute $compute_gpu_opt $ivector_opts "$modified_bnf_nnet" "$feats" ark:- \| \
    copy-feats --compress=$compress ark:- ark,scp:$bnfdir/raw_bnfeat_$name.JOB.ark,$bnfdir/raw_bnfeat_$name.JOB.scp || exit 1;
fi


N0=$(cat $data/feats.scp | wc -l)
N1=$(cat $bnfdir/raw_bnfeat_$name.*.scp | wc -l)
if [[ "$N0" != "$N1" ]]; then
  echo "$0: Error generating BNF features for $name (original:$N0 utterances, BNF:$N1 utterances)"
  exit 1;
fi

# Concatenate feats.scp into bnf_data
for n in $(seq $nj); do  cat $bnfdir/raw_bnfeat_$name.$n.scp; done > $bnf_data/feats.scp

for f in segments spk2utt text utt2spk wav.scp char.stm glm kws reco2file_and_channel stm; do
  [ -e $data/$f ] && cp -r $data/$f $bnf_data/$f
done

echo "$0: computing CMVN stats."
steps/compute_cmvn_stats.sh $bnf_data

echo "$0: done making BNF features."

exit 0;
