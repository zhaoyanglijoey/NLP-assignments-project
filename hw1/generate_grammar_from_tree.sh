#!/usr/bin/env bash

TREE=text/moresentences_devset.trees
S1_GR=grammars/devset_more_s1.gr
S2_GR=grammars/devset_more_s2.gr
VOCAB_GR=grammars/devset_more_vocab.gr
S2_UNIGRAM_GR=grammars/devset_more_s2_unigram.gr

echo "Parsing tree"
python parse_tree.py -i $TREE  -os1 $S1_GR -ov $VOCAB_GR

echo "Adding unseen words"
python vocab_generator.py -v $VOCAB_GR -a allowed_words.txt

echo "Generating s2"
python s2.py -v $VOCAB_GR -tree $TREE -s2 $S2_GR

echo "Generating unigram s2"
python s2_unigram.py -v $VOCAB_GR -s2 $S2_UNIGRAM_GR