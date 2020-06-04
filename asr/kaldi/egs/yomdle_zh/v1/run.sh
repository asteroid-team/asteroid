#!/usr/bin/env bash

set -e
stage=0
nj=60

database_slam=/export/corpora5/slam/SLAM/Chinese/transcribed
database_yomdle=/export/corpora5/slam/YOMDLE/final_chinese
download_dir=data_yomdle_chinese/download/
extra_lm=download/extra_lm.txt
data_dir=data_yomdle_chinese
exp_dir=exp_yomdle_chinese

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if [ $stage -le -1 ]; then
    local/create_download.sh --database-slam $database_slam \
        --database-yomdle $database_yomdle \
        --slam-dir download/slam_chinese \
        --yomdle-dir download/yomdle_chinese
fi

if [ $stage -le 0 ]; then
    mkdir -p data_slam_chinese/slam
    mkdir -p data_yomdle_chinese/yomdle
    local/process_data.py download/slam_chinese data_slam_chinese/slam
    local/process_data.py download/yomdle_chinese data_yomdle_chinese/yomdle
    ln -s ../data_slam_chinese/slam ${data_dir}/test
    ln -s ../data_yomdle_chinese/yomdle ${data_dir}/train
    image/fix_data_dir.sh ${data_dir}/test
    image/fix_data_dir.sh ${data_dir}/train
fi

mkdir -p $data_dir/{train,test}/data
if [ $stage -le 1 ]; then
    echo "$0: Obtaining image groups. calling get_image2num_frames"
    echo "Date: $(date)."
    image/get_image2num_frames.py --feat-dim 60 $data_dir/train
    image/get_allowed_lengths.py --frame-subsampling-factor 4 10 $data_dir/train

    for datasplit in train test; do
        echo "$0: Extracting features and calling compute_cmvn_stats for dataset: $datasplit. "
        echo "Date: $(date)."
        local/extract_features.sh --nj $nj --cmd "$cmd" \
            --feat-dim 60 --num-channels 3 \
            $data_dir/${datasplit}
        steps/compute_cmvn_stats.sh $data_dir/${datasplit} || exit 1;
    done

    echo "$0: Fixing data directory for train dataset"
    echo "Date: $(date)."
    utils/fix_data_dir.sh $data_dir/train
fi

if [ $stage -le 2 ]; then
    for datasplit in train; do
        echo "$(date) stage 2: Performing augmentation, it will double training data"
        local/augment_data.sh --nj $nj --cmd "$cmd" --feat-dim 60 $data_dir/${datasplit} $data_dir/${datasplit}_aug $data_dir
        steps/compute_cmvn_stats.sh $data_dir/${datasplit}_aug || exit 1;
    done
fi

if [ $stage -le 3 ]; then
    echo "$0: Preparing dictionary and lang..."
    if [ ! -f $data_dir/train/bpe.out ]; then
        cut -d' ' -f2- $data_dir/train/text | utils/lang/bpe/prepend_words.py | python3 utils/lang/bpe/learn_bpe.py -s 700 > $data_dir/train/bpe.out
        for datasplit in test train train_aug; do
            cut -d' ' -f1 $data_dir/$datasplit/text > $data_dir/$datasplit/ids
            cut -d' ' -f2- $data_dir/$datasplit/text | utils/lang/bpe/prepend_words.py | python3 utils/lang/bpe/apply_bpe.py -c $data_dir/train/bpe.out | sed 's/@@//g' > $data_dir/$datasplit/bpe_text
            mv $data_dir/$datasplit/text $data_dir/$datasplit/text.old
            paste -d' ' $data_dir/$datasplit/ids $data_dir/$datasplit/bpe_text > $data_dir/$datasplit/text
        done
    fi

    local/prepare_dict.sh --data-dir $data_dir --dir $data_dir/local/dict
    # This recipe uses byte-pair encoding, the silences are part of the words' pronunciations.
    # So we set --sil-prob to 0.0
    utils/prepare_lang.sh --num-sil-states 4 --num-nonsil-states 6 --sil-prob 0.0 --position-dependent-phones false \
        $data_dir/local/dict "<sil>" $data_dir/lang/temp $data_dir/lang
    silphonelist=`cat $data_dir/lang/phones/silence.csl`
    nonsilphonelist=`cat $data_dir/lang/phones/nonsilence.csl`
    local/gen_topo.py 8 4 10 $nonsilphonelist $silphonelist $data_dir/lang/phones.txt > $data_dir/lang/topo 
    utils/lang/bpe/add_final_optional_silence.sh --final-sil-prob 0.5 $data_dir/lang
fi

if [ $stage -le 4 ]; then
    echo "$0: Estimating a language model for decoding..."
    local/train_lm.sh --data-dir $data_dir  --dir $data_dir/local/local_lm
    utils/format_lm.sh $data_dir/lang $data_dir/local/local_lm/data/arpa/3gram_unpruned.arpa.gz \
        $data_dir/local/dict/lexicon.txt $data_dir/lang_test
fi

if [ $stage -le 5 ]; then
    echo "$0: Calling the flat-start chain recipe..."
    echo "Date: $(date)." 
    local/chain/run_flatstart_cnn1a.sh --nj $nj --train-set train_aug --data-dir $data_dir --exp-dir $exp_dir
fi

if [ $stage -le 6 ]; then
    echo "$0: Aligning the training data using the e2e chain model..."
    echo "Date: $(date)."
    steps/nnet3/align.sh --nj $nj --cmd "$cmd" --use-gpu false \
        --scale-opts '--transition-scale=1.0 --acoustic-scale=1.0 --self-loop-scale=1.0' \
        $data_dir/train_aug $data_dir/lang $exp_dir/chain/e2e_cnn_1a $exp_dir/chain/e2e_ali_train
fi

if [ $stage -le 7 ]; then
    echo "$0: Building a tree and training a regular chain model using the e2e alignments..."
    echo "Date: $(date)."
    local/chain/run_cnn_e2eali_1b.sh --nj $nj --train-set train_aug --data-dir $data_dir --exp-dir $exp_dir
fi

if [ $stage -le 8 ]; then
    echo "$0: Estimating a language model for lattice rescoring...$(date)"
    local/train_lm_lr.sh --data-dir $data_dir  --dir $data_dir/local/local_lm_lr --extra-lm $extra_lm --order 6

    utils/build_const_arpa_lm.sh $data_dir/local/local_lm_lr/data/arpa/6gram_unpruned.arpa.gz \
        $data_dir/lang_test $data_dir/lang_test_lr
    steps/lmrescore_const_arpa.sh $data_dir/lang_test $data_dir/lang_test_lr \
        $data_dir/test $exp_dir/chain/cnn_e2eali_1b/decode_test $exp_dir/chain/cnn_e2eali_1b/decode_test_lr
fi
