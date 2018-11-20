import argparse, sys, os, logging
from itertools import islice
import pickle
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import math
import matplotlib.pyplot as plt
from HMMmodel import HMMmodel

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-d", "--datadir", dest="datadir", default="data", help="data directory (default=data)")
    argparser.add_argument("-p", "--prefix", dest="fileprefix", default="hansards", help="prefix of parallel data files (default=hansards)")
    argparser.add_argument("-e", "--english", dest="english", default="en", help="suffix of English (target language) filename (default=en)")
    argparser.add_argument("-f", "--french", dest="french", default="fr", help="suffix of French (source language) filename (default=fr)")
    argparser.add_argument("-l", "--logfile", dest="logfile", default=None, help="filename for logging output")
    argparser.add_argument("-t", "--threshold", dest="threshold", default=0.5, type=float, help="threshold for alignment (default=0.5)")
    argparser.add_argument("-n", "--num_sentences", dest="num_sents", default=sys.maxsize, type=int, help="Number of sentences to use for training and alignment")
    argparser.add_argument('-r', '--resume', default=None, help='resume training')
    argparser.add_argument("--epsilon", dest="epsilon", default=1, type=float, help="Convergence check passes if |L(t_k)-L(t_k-1)|<epsilon")
    argparser.add_argument("--max-iteration", dest="max_iteration", default=10, type=int, help="max number of iteration")
    # argparser.add_argument('--iter', type=int, default=100)
    argparser.add_argument("--save-model", dest="save_model", default="hmmmodel.pickle", help="save variable t")
    argparser.add_argument("--load-model", dest="load_model", help="model file of variable t")
    argparser.add_argument('--ckptdir', default='hmmckpt', help='check point dir')
    argparser.add_argument("-a", "--alignments", dest="alignment", default="a",
                         help="suffix of gold alignments filename (default=a)")
    args = argparser.parse_args()
    f_data = "%s.%s" % (os.path.join(args.datadir, args.fileprefix), args.french)
    e_data = "%s.%s" % (os.path.join(args.datadir, args.fileprefix), args.english)
    a_data = "%s.%s" % (os.path.join(args.datadir, args.fileprefix), args.alignment)
    with open(f_data) as f, open(e_data) as e, open(a_data) as a:
        f_data, e_data, a_data = f.readlines()[:args.num_sents],\
                                 e.readlines()[:args.num_sents], \
                                 a.readlines()[:args.num_sents]

    if args.logfile:
            logging.basicConfig(filename=args.logfile, filemode='w', level=logging.INFO)

    bitext = [[sentence.strip().split() for sentence in pair] for pair in islice(
        zip(f_data, e_data), 0, args.num_sents)]
    hmmmodel = HMMmodel()
    if args.load_model:
        hmmmodel.load_model(args.load_model)
    else:
        if not os.path.exists(args.ckptdir):
            os.mkdir(args.ckptdir)

        if args.resume:
            hmmmodel.load_model(args.resume)
            hmmmodel.train(bitext, args.max_iteration,
                      args.ckptdir, f_data, e_data, a_data, args.epsilon)
        else:
            hmmmodel.init_params(bitext)
            hmmmodel.train(bitext, args.max_iteration,
                      args.ckptdir, f_data, e_data, a_data, args.epsilon)
        hmmmodel.dump_model(args.save_model)
    for f_sentence, e_sentence in bitext:
        J = len(f_sentence)
        alignments, _ = hmmmodel.viterbi_decode(f_sentence, e_sentence)
        for j in range(J):
            print("{0}-{1}".format(j, alignments[j]), end=" ")
        print()



if __name__ == '__main__':
    main()