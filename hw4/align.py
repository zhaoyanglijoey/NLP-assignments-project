#!/usr/bin/env python
import optparse, sys, os, logging
from collections import defaultdict
from itertools import islice

def build_vocab(bitext):
  sys.stderr.write("Building vocab...\n")
  f_vocab = set()
  e_vocab = set()
  for f, e in bitext:
    f_vocab = f_vocab | set(f)
    e_vocab = e_vocab | set(e)
  return (f_vocab, e_vocab)

if __name__ == '__main__':

  optparser = optparse.OptionParser()
  optparser.add_option("-d", "--datadir", dest="datadir", default="data", help="data directory (default=data)")
  optparser.add_option("-p", "--prefix", dest="fileprefix", default="hansards", help="prefix of parallel data files (default=hansards)")
  optparser.add_option("-e", "--english", dest="english", default="en", help="suffix of English (target language) filename (default=en)")
  optparser.add_option("-f", "--french", dest="french", default="fr", help="suffix of French (source language) filename (default=fr)")
  optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="filename for logging output")
  optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="threshold for alignment (default=0.5)")
  optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxsize, type="int", help="Number of sentences to use for training and alignment")
  (opts, _) = optparser.parse_args()
  f_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.french)
  e_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.english)

  if opts.logfile:
      logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.INFO)

  bitext = [[sentence.strip().split() for sentence in pair] for pair in islice(zip(open(f_data), open(e_data)), opts.num_sents)]

  (f_vocab, e_vocab) = build_vocab(bitext)
  t0 = 1/len(f_vocab)

  sys.stderr.write("Training...\n")

  t = defaultdict(float)

  for f_word in f_vocab:
    for e_word in e_vocab:
      t[(f_word, e_word)] = t0
