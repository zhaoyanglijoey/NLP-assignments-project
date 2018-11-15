import argparse, sys, os, logging
from itertools import islice
import pickle
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import math

def dump_model(file, iter, pr_trans, pr_emit, pr_prior):
    model = {
        'iter': iter,
        'pr_trans': pr_trans,
        'pr_emit': pr_emit,
        'pr_prior': pr_prior
    }
    with open(file, 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

def load_model(file):
    with open(file, 'rb') as f:
        model = pickle.load(file)
    iter = model['iter']
    pr_trans = model['pr_trans']
    pr_emit = model['pr_emit']
    pr_prior = model['pr_prior']

    return iter, pr_trans, pr_emit, pr_prior

def build_vocab(bitext):
    sys.stderr.write("Building vocab...\n")
    f_list = []
    e_list = []
    for f, e in bitext:
        f_list += f
        e_list += e
    f_vocab = set(f_list)
    e_vocab = set(e_list)
    sys.stderr.write('done\n')
    return (f_vocab, e_vocab)

def init_params(bitext, f_vocab_size):
    pr_trans = {}
    pr_emit = {}
    pr_prior = {}


    # maxe_len = 0
    for f_sentence, e_sentence in bitext:
        I = len(e_sentence)
        J = len(f_sentence)
        # maxe_len = max(maxe_len, I)
        for j, f in enumerate(f_sentence):
            for i, e in enumerate(e_sentence):
                pr_emit[(f, e)] = 1 / f_vocab_size
        for i in range(I):
            pr_prior[(i, I)] = 1 / I
            for i_p in range(I):
                pr_trans[(i, i_p, I)] = 1 / I
    # for i in range(maxe_len):
    #     pr_prior[i] = 1 / (maxe_len+1)

    return pr_trans, pr_emit, pr_prior

def forward_backford(f_sentence, e_sentence, pr_trans, pr_emit, pr_prior):
    I = len(e_sentence)
    J = len(f_sentence)

    forward_pr = np.zeros((I, J), dtype=float)
    backward_pr = np.zeros((I, J), dtype=float)

    for i in range(I):
        forward_pr[i][0] = pr_prior[(i, I)] * pr_emit[(f_sentence[0], e_sentence[i])]

    for j in range(1, J):
        for i in range(I):
            trans = 0
            for i_p in range(I):
                trans += forward_pr[i_p][j-1] * pr_trans[(i, i_p, I)]
            forward_pr[i][j] = pr_emit[(f_sentence[j], e_sentence[i])] * trans

    for i in range(I):
        backward_pr[i][J-1] = 1

    for j in range(J-1)[::-1]:
        for i_p in range(I):
            tmp = 0
            for i in range(I):
                tmp += backward_pr[i][j+1] * pr_trans[(i, i_p, I)] * pr_emit[(f_sentence[j+1], e_sentence[i])]
            backward_pr[i_p][j] = tmp

    return forward_pr, backward_pr


def train(iter, pr_trans, pr_emit, pr_prior, bitext, max_iteration, ckpt, epsilon = 0.01 ):
    e_lens = set()

    for f_sentence, e_sentence in bitext:
        e_lens.add(len(e_sentence))

    prev_llh = calc_llh(bitext, pr_trans, pr_emit, pr_prior)
    sys.stderr.write('iteration {}, llh {}\n'.format(iter, prev_llh))

    while iter < max_iteration:
        iter += 1
        sys.stderr.write('training iter {}...\n'.format(iter))
        c_emit = defaultdict(float)
        c_trans = defaultdict(float)
        c_emit_margin = defaultdict(float)
        c_trans_margin = defaultdict(float)
        c_prior = defaultdict(float)
        c_prior_margin = defaultdict(float)
        for f_sentence, e_sentence in tqdm(bitext):
            I = len(e_sentence)
            J = len(f_sentence)
            forward_pr, backword_pr = forward_backford(f_sentence, e_sentence, pr_trans, pr_emit, pr_prior)
            denominator = np.sum(forward_pr[:, J-1])
            for i in range(I):
                for j in range(J):
                    gamma = forward_pr[i][j] * backword_pr[i][j] / denominator
                    c_emit[(f_sentence[j], e_sentence[i])] += gamma
                    c_emit_margin[e_sentence[i]] += gamma
                    if j == 0:
                        c_prior[(i, I)] += gamma
                        c_prior_margin[I] += gamma

            for j in range(J-1):
                for i in range(I):
                    for i_p in range(I):
                        si = forward_pr[i_p][j] * pr_trans[(i, i_p, I)] * backword_pr[i][j+1] * \
                             pr_emit[(f_sentence[j+1], e_sentence[i])] / denominator
                        c_trans[(i-i_p, I)] += si
                        c_trans_margin[i_p] += si
        for f_sentence, e_sentence in bitext:
            I = len(e_sentence)
            J = len(f_sentence)
            for f in f_sentence:
                for e in e_sentence:
                    pr_emit[(f, e)] = c_emit[(f, e)] / c_emit_margin[e]
        for I in e_lens:
            for i in range(I):
                for i_p in range(I):
                    pr_trans[(i, i_p, I)] = c_trans[(i-i_p, I)] / c_trans_margin[i_p]
            for i in range(I):
                pr_prior[(i, I)] = c_prior[(i, I)] / c_prior_margin[I]

        dump_model(ckpt, iter, pr_trans, pr_emit, pr_prior)

        llh = calc_llh(bitext, pr_trans, pr_emit, pr_prior)
        sys.stderr.write('iteration {}, llh {}\n'.format(iter, llh))

        if abs(llh - prev_llh) < epsilon:
            break

        prev_llh = llh

    return iter, pr_trans, pr_emit, pr_prior

def viterbi_decode(f_sentence, e_sentence, pr_trans, pr_emit, pr_prior):
    I = len(e_sentence)
    J = len(f_sentence)
    V = np.zeros((I, J), dtype=float)
    backptr = np.zeros((I, J), dtype=int)
    for i in range(I):
        V[i][0] = math.log(pr_prior[(i, I)]) + math.log(pr_emit[(f_sentence[0], e_sentence[i])])
    for j in range(1, J):
        for i in range(I):
            for i_p in range(I):
                tmp = V[i_p][j-1] + math.log(pr_trans[(i, i_p, I)]) +  \
                              math.log(pr_emit[(f_sentence[j], e_sentence[i])])
                if tmp > V[i][j]:
                    V[i][j] = tmp
                    backptr[i][j] = i_p
    best_idx = np.argmax(V[:, J-1])
    best_score = V[best_idx, J-1]

    alignments = [best_idx]
    for j in range(1, J)[::-1]:
        best_idx = backptr[best_idx, j]
        alignments.append(best_idx)

    alignments.reverse()
    return (alignments, best_score)

def calc_llh(bitext, pr_trans, pr_emit, pr_prior):
    sys.stderr.write('calculating log likelyhood...\n')
    llh = 0
    for f_sentence, e_sentence in tqdm(bitext):
        llh += viterbi_decode(f_sentence, e_sentence, pr_trans, pr_emit, pr_prior)[1]
    sys.stderr.write('done\n')
    return llh

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-d", "--datadir", dest="datadir", default="data", help="data directory (default=data)")
    argparser.add_argument("-p", "--prefix", dest="fileprefix", default="hansards", help="prefix of parallel data files (default=hansards)")
    argparser.add_argument("-e", "--english", dest="english", default="en", help="suffix of English (target language) filename (default=en)")
    argparser.add_argument("-f", "--french", dest="french", default="fr", help="suffix of French (source language) filename (default=fr)")
    argparser.add_argument("-l", "--logfile", dest="logfile", default=None, help="filename for logging output")
    argparser.add_argument("-t", "--threshold", dest="threshold", default=0.5, type=float, help="threshold for alignment (default=0.5)")
    argparser.add_argument("-n", "--num_sentences", dest="num_sents", default=sys.maxsize, type=int, help="Number of sentences to use for training and alignment")
    argparser.add_argument("--epsilon", dest="epsilon", default=1, type=float, help="Convergence check passes if |L(t_k)-L(t_k-1)|<epsilon")
    argparser.add_argument("--max-iteration", dest="max_iteration", default=1e-3, type=int, help="max number of iteration")
    argparser.add_argument("--save-model", dest="save_model", default="ibm_model_i.pickle", help="save variable t")
    argparser.add_argument("--load-model", dest="load_model", help="model file of variable t")
    argparser.add_argument('-r', '--resume', help='resume training')
    argparser.add_argument('--ckpt', default='ckpt.pickle', help='check point')
    args = argparser.parse_args()
    f_data = "%s.%s" % (os.path.join(args.datadir, args.fileprefix), args.french)
    e_data = "%s.%s" % (os.path.join(args.datadir, args.fileprefix), args.english)

    if args.logfile:
            logging.basicConfig(filename=args.logfile, filemode='w', level=logging.INFO)

    bitext = [[sentence.strip().split() for sentence in pair] for pair in islice(zip(open(f_data), open(e_data)), args.num_sents)]

    if args.load_model:
        iter, pr_trans, pr_emit, pr_prior = load_model(args.load_model)
    else:
        if args.resume:
            iter, pr_trans, pr_emit, pr_prior = load_model(args.resume)
            iter, pr_trans, pr_emit, pr_prior = \
                train(iter, pr_trans, pr_emit, pr_prior, bitext, args.max_iteration, args.ckpt, args.epsilon)
        else:
            f_vocab, e_vocab = build_vocab(bitext)
            pr_trans, pr_emit, pr_prior = init_params(bitext, len(f_vocab))
            iter, pr_trans, pr_emit, pr_prior = \
                train(0, pr_trans, pr_emit, pr_prior, bitext, args.max_iteration, args.ckpt, args.epsilon)

    for f_sentence, e_sentence in bitext:
        J = len(f_sentence)
        alignments, _ = viterbi_decode(f_sentence, e_sentence, pr_trans, pr_emit, pr_prior)
        for j in range(J):
            print("{0}-{1}".format(j, alignments[j]), end=" ")
        print()



if __name__ == '__main__':
    main()