#!/usr/bin/env bash

TREE=train.trees
S1_GR=train_s1.gr
S2_GR=train_s2.gr
VOCAB_GR=train_vocab.gr

echo "Parsing tree"
python parse_tree.py -i $TREE -os1 $S1_GR -ov $VOCAB_GR

echo "Adding unseen words"
python vocab_generator.py -v $VOCAB_GR -a allowed_words.txt

echo "Generating s2"
python s2.py -v $VOCAB_GR -s2 $S2_GR