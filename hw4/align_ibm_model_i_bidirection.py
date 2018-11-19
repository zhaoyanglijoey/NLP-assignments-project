#!/usr/bin/env python
import optparse, sys, os, logging
from itertools import islice
import pickle
from tqdm import tqdm
from ibmmodel1 import ibm_model_1


def build_vocab(bitext):
    sys.stderr.write("Building vocab...\n")
    f_list = []
    e_list = []
    for f, e in tqdm(bitext):
        f_list += f
        e_list += e
    f_vocab = set(f_list)
    e_vocab = set(e_list)
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
    optparser.add_option("--epsilon", dest="epsilon", default=0.0001, type="float", help="Convergence check passes if |L(t_k)-L(t_k-1)|<epsilon")
    optparser.add_option("--max-iteration", dest="max_iteration", default=100, type="int", help="max number of iteration")
    optparser.add_option("--save-model", dest="save_model", default="ibm_model_i_bidirection.pickle", help="save variable t")
    optparser.add_option("--load-model", dest="load_model", help="model file of variable t")
    (opts, _) = optparser.parse_args()
    f_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.french)
    e_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.english)

    if opts.logfile:
            logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.INFO)

    bitext = [[sentence.strip().split() for sentence in pair] for pair in islice(zip(open(f_data), open(e_data)), opts.num_sents)]

    (f_vocab, e_vocab) = build_vocab(bitext)
    load_model = opts.load_model
    if load_model:
        sys.stderr.write("Use model {}...\n".format(load_model))
        with open(load_model, 'rb') as f:
            [t_1, t_2] = pickle.load(f)
        for sentence_pair in bitext:
            sentence_pair.reverse()
    else:
        t_1 = ibm_model_1.train(bitext, f_vocab, e_vocab, opts.max_iteration, opts.epsilon)
        for sentence_pair in bitext:
            sentence_pair.reverse()
        t_2 = ibm_model_1.train(bitext, e_vocab, f_vocab, opts.max_iteration, opts.epsilon)

        # save model file
        with open(opts.save_model, 'wb') as f:
            pickle.dump([t_1, t_2], f)
            sys.stderr.write("Saved model to file {}\n".format(opts.save_model))

    alignments_list_2 = ibm_model_1.decode(bitext, t_2)
    for sentence_pair in bitext:
        sentence_pair.reverse()
    alignments_list_1 = ibm_model_1.decode(bitext, t_1)
    # print alignments
    alignments_list = ibm_model_1.alignments_intersection(alignments_list_1, alignments_list_2)
    for alignments in alignments_list:
        for i, j in alignments:
            print("{0}-{1}".format(i, j), end=" ")
        print()
