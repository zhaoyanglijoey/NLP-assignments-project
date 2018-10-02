#!/usr/bin/env bash

TREE=text/dev_and_example.trees
S1_GR=grammars/dev_and_exp_s1.gr
S2_GR=grammars/dev_and_exp_s2.gr
VOCAB_GR=grammars/dev_and_exp_vocab.gr

echo "Parsing tree"
python parse_tree.py -i $TREE -os1 $S1_GR -ov $VOCAB_GR

echo "Adding unseen words"
python vocab_generator.py -v $VOCAB_GR -a allowed_words.txt

echo "Generating s2"
python s2.py -v $VOCAB_GR -tree $TREE -s2 $S2_GR