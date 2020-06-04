#!/usr/bin/env bash

# Copyright (C) 2016, Qatar Computing Research Institute, HBKU

if [ -f path.sh ]; then
  . path.sh; else
   echo "missing path.sh"; exit 1;
fi 

lang_test=data/lang_test
arpa_lm=data/local/lm/3gram-mincount/lm_unpruned.gz

. utils/parse_options.sh

mkdir -p $lang_test

[ ! -f $arpa_lm ] && echo No such file $arpa_lm && exit 1;

rm -r $lang_test
cp -r data/lang $lang_test

# grep -v '<s> <s>' etc. is only for future-proofing this script.  Our
# LM doesn't have these "invalid combinations".  These can cause 
# determinization failures of CLG [ends up being epsilon cycles].
# Note: remove_oovs.pl takes a list of words in the LM that aren't in
# our word list.  Since our LM doesn't have any, we just give it
# /dev/null [we leave it in the script to show how you'd do it].
gunzip -c "$arpa_lm" | \
   grep -v '<s> <s>' | \
   grep -v '</s> <s>' | \
   grep -v '</s> </s>' | \
   arpa2fst - | fstprint | \
   utils/remove_oovs.pl /dev/null | \
   utils/eps2disambig.pl | utils/s2eps.pl | fstcompile --isymbols=$lang_test/words.txt \
     --osymbols=$lang_test/words.txt  --keep_isymbols=false --keep_osymbols=false | \
    fstrmepsilon | fstarcsort --sort_type=ilabel > $lang_test/G.fst
  fstisstochastic $lang_test/G.fst


echo  "Checking how stochastic G is (the first of these numbers should be small):"
fstisstochastic $lang_test/G.fst 

## Check lexicon.
## just have a look and make sure it seems sane.
echo "First few lines of lexicon FST:"
fstprint   --isymbols=data/lang/phones.txt --osymbols=data/lang/words.txt data/lang/L.fst  | head

echo Performing further checks

# Checking that G.fst is determinizable.
fstdeterminize $lang_test/G.fst /dev/null || echo Error determinizing G.

# Checking that L_disambig.fst is determinizable.
fstdeterminize $lang_test/L_disambig.fst /dev/null || echo Error determinizing L.

# Checking that disambiguated lexicon times G is determinizable
# Note: we do this with fstdeterminizestar not fstdeterminize, as
# fstdeterminize was taking forever (presumbaly relates to a bug
# in this version of OpenFst that makes determinization slow for
# some case).
fsttablecompose $lang_test/L_disambig.fst $lang_test/G.fst | \
   fstdeterminizestar >/dev/null || echo Error

# Checking that LG is stochastic:
fsttablecompose data/lang/L_disambig.fst $lang_test/G.fst | \
   fstisstochastic || echo LG is not stochastic


echo mgb_format_data  succeeded.
