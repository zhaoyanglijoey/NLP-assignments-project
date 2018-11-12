#!/usr/bin/env python
import optparse, sys, os, logging
from collections import defaultdict
from itertools import islice
import math

def build_vocab(bitext):
  sys.stderr.write("Building vocab...\n")
  f_vocab = set()
  e_vocab = set()
  for f, e in bitext:
    f_vocab = f_vocab | set(f)
    e_vocab = e_vocab | set(e)
  return (f_vocab, e_vocab)

def calculate_llh(bitext, t):
  llh = 0
  for f, e in bitext:
    for f_word in f:
      t_sum = 0
      for e_word in e:
        t_sum += t[(f_word, e_word)]
      llh += math.log(t_sum)
  return llh

if __name__ == '__main__':

  optparser = optparse.OptionParser()
  optparser.add_option("-d", "--datadir", dest="datadir", default="data", help="data directory (default=data)")
  optparser.add_option("-p", "--prefix", dest="fileprefix", default="hansards", help="prefix of parallel data files (default=hansards)")
  optparser.add_option("-e", "--english", dest="english", default="en", help="suffix of English (target language) filename (default=en)")
  optparser.add_option("-f", "--french", dest="french", default="fr", help="suffix of French (source language) filename (default=fr)")
  optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="filename for logging output")
  optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="threshold for alignment (default=0.5)")
  optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxsize, type="int", help="Number of sentences to use for training and alignment")
  optparser.add_option("--epsilon", dest="epsilon", default=0.0001, type="float", help="Convergence check passes if |L(t_k)-L(t_k-1)|<epsilon")
  optparser.add_option("--max-iteration", dest="max_iteration", default=100, type="int", help="max number of iteration")
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

  llh_old = calculate_llh(bitext, t)
  k = 0
  while True:
    k += 1
    if k > opts.max_iteration:
      sys.stderr.write("Training finished.\n")
      break

    # Training
    count_pair = defaultdict(float)
    count_e = defaultdict(float)
    for f, e in bitext:
      for f_word in f:
        z = 0
        for e_word in e:
          z += t[(f_word, e_word)]
        for e_word in e:
          c = t[(f_word, e_word)] / z
          count_pair[(f_word, e_word)] += c
          count_e[e_word] += c
    for f_word, e_word in count_pair:
      t[(f_word, e_word)] = count_pair[(f_word, e_word)] / count_e[e_word]

    # Calculate log likelihood
    llh = calculate_llh(bitext, t)
    sys.stderr.write("Log likelihood after iteration {0}: {1}\n".format(k, llh))

    if abs(llh - llh_old) < opts.epsilon:
      sys.stderr.write("Training finished.\n")
      break

    llh_old = llh
