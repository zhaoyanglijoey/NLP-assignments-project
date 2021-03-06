import matplotlib
matplotlib.use('Agg')
import argparse, sys, os, logging
from itertools import islice
import pickle
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import math
import matplotlib.pyplot as plt
from HMMmodel import BiHMMmodel, score_alignments

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
    argparser.add_argument("--iter", dest="iter", default=5, type=int, help="number of iteration")
    argparser.add_argument("--save-model", dest="save_model", default="bihmm.m", help="save variable t")
    argparser.add_argument("--load-model", dest="load_model", help="model file of variable t")
    argparser.add_argument('--loadibm1')
    argparser.add_argument('--ckptdir', default='bihmmckpt', help='check point dir')
    argparser.add_argument("-a", "--alignments", dest="alignment", default="a",
                         help="suffix of gold alignments filename (default=a)")
    args = argparser.parse_args()
    f_data = "%s.%s" % (os.path.join(args.datadir, args.fileprefix), args.french)
    e_data = "%s.%s" % (os.path.join(args.datadir, args.fileprefix), args.english)
    a_data = "%s.%s" % (os.path.join(args.datadir, args.fileprefix), args.alignment)
    validate = False
    if args.fileprefix != 'hansards':
        validate = False
        with open(f_data) as f, open(e_data) as e:
            f_data, e_data = f.readlines(), e.readlines()
            a_data = None
    else:
        with open(f_data) as f, open(e_data) as e, open(a_data) as a:
            f_data, e_data, a_data = f.readlines()[:args.num_sents],\
                                 e.readlines()[:args.num_sents], \
                                 a.readlines()[:args.num_sents]

    if args.logfile:
            logging.basicConfig(filename=args.logfile, filemode='w', level=logging.INFO)

    bitext = [[sentence.strip().split() for sentence in pair] for pair in islice(
        zip(f_data, e_data), 0, args.num_sents)]
    rev_bitext = [[e_sentence, f_setence] for f_setence, e_sentence in bitext]

    bihmmmodel = BiHMMmodel()
    if args.load_model:
        bihmmmodel.load_model(args.load_model)
    else:
        if not os.path.exists(args.ckptdir):
            os.mkdir(args.ckptdir)

        if args.resume:
            # resume training from previouly saved model
            bihmmmodel.load_model(args.resume)
            bihmmmodel.train(bitext, rev_bitext, args.iter,
                      args.ckptdir, f_data, e_data, None, validate)
        else:
            # start training from scratch
            bihmmmodel.init_params(bitext, rev_bitext)
            if args.loadibm1:
                # load pretrained ibm1 model parameters
                bihmmmodel.load_from_ibm1(args.loadibm1)
                if validate:
                    # computer AER score
                    bihmmmodel.validate(bitext, rev_bitext, f_data, e_data, a_data)
            bihmmmodel.train(bitext, rev_bitext, args.iter,
                      args.ckptdir, f_data, e_data, None, validate)
        bihmmmodel.dump_model(args.save_model)

    if validate:
        bihmmmodel.validate(bitext, rev_bitext, f_data, e_data, a_data)

    alignments_list = bihmmmodel.decode(bitext, rev_bitext)
    for alignments in alignments_list:
        for i, j in alignments:
            print("{0}-{1}".format(i, j), end=" ")
        print()




if __name__ == '__main__':
    main()