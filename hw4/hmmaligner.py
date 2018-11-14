import argparse, sys, os, logging
from itertools import islice
import pickle
from tqdm import tqdm

def init_params(bitext):
    pr_trans = {}
    pr_emit = {}

    for f_sentence, e_sentence in bitext:
        for j, f in enumerate(f_sentence):
            for i, e in enumerate(e_sentence):
                pr_emit[(f, e)] = 1

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-d", "--datadir", dest="datadir", default="data", help="data directory (default=data)")
    argparser.add_argument("-p", "--prefix", dest="fileprefix", default="hansards", help="prefix of parallel data files (default=hansards)")
    argparser.add_argument("-e", "--english", dest="english", default="en", help="suffix of English (target language) filename (default=en)")
    argparser.add_argument("-f", "--french", dest="french", default="fr", help="suffix of French (source language) filename (default=fr)")
    argparser.add_argument("-l", "--logfile", dest="logfile", default=None, help="filename for logging output")
    argparser.add_argument("-t", "--threshold", dest="threshold", default=0.5, type="float", help="threshold for alignment (default=0.5)")
    argparser.add_argument("-n", "--num_sentences", dest="num_sents", default=sys.maxsize, type="int", help="Number of sentences to use for training and alignment")
    argparser.add_argument("--epsilon", dest="epsilon", default=0.0001, type="float", help="Convergence check passes if |L(t_k)-L(t_k-1)|<epsilon")
    argparser.add_argument("--max-iteration", dest="max_iteration", default=100, type="int", help="max number of iteration")
    argparser.add_argument("--save-model", dest="save_model", default="ibm_model_i.pickle", help="save variable t")
    argparser.add_argument("--load-model", dest="load_model", help="model file of variable t")
    opts = argparser.parse_args()
    f_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.french)
    e_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.english)

    if opts.logfile:
            logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.INFO)

    bitext = [[sentence.strip().split() for sentence in pair] for pair in islice(zip(open(f_data), open(e_data)), opts.num_sents)]



if __name__ == '__main__':
    main()