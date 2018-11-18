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
        model = pickle.load(f)
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

def forward_backward(f_sentence, e_sentence, pr_trans, pr_emit, pr_prior, scale):
    I = len(e_sentence)
    J = len(f_sentence)

    # forward_pr = np.zeros((I, J), dtype=np.float64)
    # backward_pr = np.zeros((I, J), dtype=np.float64)
    forward_pr = [[0.]*J for _ in range(I)]
    backward_pr = [[0.]*J for _ in range(I)]

    for i in range(I):
        forward_pr[i][0] = pr_prior[(i, I)] * scale * pr_emit[(f_sentence[0], e_sentence[i])] * scale

    for j in range(1, J):
        for i in range(I):
            trans = 0
            for i_p in range(I):
                trans += forward_pr[i_p][j-1] * scale * pr_trans[(i, i_p, I)] * scale
            forward_pr[i][j] = pr_emit[(f_sentence[j], e_sentence[i])] * scale * trans

    for i in range(I):
        backward_pr[i][J-1] = 1

    for j in range(J-1)[::-1]:
        for i_p in range(I):
            tmp = 0
            for i in range(I):
                tmp += backward_pr[i][j+1] * pr_trans[(i, i_p, I)] * scale * pr_emit[(f_sentence[j+1], e_sentence[i])] * scale
            backward_pr[i_p][j] = tmp

    return forward_pr, backward_pr


def train(iter, pr_trans, pr_emit, pr_prior, bitext, max_iteration, ckpt, epsilon = None):
    e_lens = set()

    for f_sentence, e_sentence in bitext:
        e_lens.add(len(e_sentence))
    f_vocab, e_vocab = build_vocab(bitext)
    f_vocab_size = len(f_vocab)
    prev_llh = calc_llh(bitext, pr_trans, pr_emit, pr_prior)
    sys.stderr.write('iteration {}, llh {}\n'.format(iter, prev_llh))

    alpha = 0.1
    beta = 0.0
    while iter < max_iteration:
        iter += 1
        sys.stderr.write('training iter {}...\n'.format(iter))
        c_emit = defaultdict(float)
        c_trans = defaultdict(float)
        c_emit_margin = defaultdict(lambda: 1e-100)
        c_trans_margin = defaultdict(lambda : 1e-100)
        c_prior = defaultdict(float)
        c_prior_margin = defaultdict(lambda: 1e-100)
        for f_sentence, e_sentence in tqdm(bitext):
            I = len(e_sentence)
            J = len(f_sentence)
            forward_pr, backword_pr = forward_backward(f_sentence, e_sentence, pr_trans, pr_emit, pr_prior, I)
            denominator = 0
            for i in range(I):
                denominator += forward_pr[i][J-1]
            if denominator == 0:
                print('0 denominator!')
            # print(denominator)
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
                        # c_trans_margin[(i_p, I)] += si
        for f_sentence, e_sentence in bitext:
            for f in f_sentence:
                for e in e_sentence:
                    pr_emit[(f, e)] = beta * (1 / f_vocab_size) + (1-beta) * (c_emit[(f, e)] / c_emit_margin[e])
        for I in e_lens:
            for i_p in range(I):
                margin = 0
                for i_pp in range(I):
                    margin += c_trans[(i_pp-i_p, I)]
                for i in range(I):
                    pr_trans[(i, i_p, I)] = alpha * 1 / I + (1-alpha) * (c_trans[(i-i_p, I)] / margin)
            for i in range(I):
                pr_prior[(i, I)] = alpha * 1 / I + (1-alpha) * (c_prior[(i, I)] / c_prior_margin[I])

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
    V = [[0.]*J for _ in range(I)]
    # V = np.zeros((I, J), dtype=float)
    backptr = [[0]*J for _ in range(I)]
    for i in range(I):
        V[i][0] = math.log(pr_prior[(i, I)]) + math.log(pr_emit[(f_sentence[0], e_sentence[i])])
        # V[i][0] = pr_prior[(i, I)] * pr_emit[(f_sentence[0], e_sentence[i])]

    for j in range(1, J):
        for i in range(I):
            for i_p in range(I):
                tmp = V[i_p][j-1] + math.log(pr_trans[(i, i_p, I)]) +  \
                              math.log(pr_emit[(f_sentence[j], e_sentence[i])])
                # tmp = V[i_p][j-1] * pr_trans[(i, i_p, I)] * pr_emit[(f_sentence[j], e_sentence[i])]
                if i_p == 0 or (i_p != 0 and tmp > V[i][j]):
                    V[i][j] = tmp
                    backptr[i][j] = i_p
    best_idx = 0
    best_score = V[best_idx][J-1]
    for i in range(1, I):
        if V[i][J-1] > best_score:
            best_score = V[i][J-1]
            best_idx = i

    alignments = [best_idx]
    for j in range(1, J)[::-1]:
        best_idx = backptr[best_idx][j]
        alignments.append(best_idx)

    alignments.reverse()
    return (alignments, best_score)

def calc_llh(bitext, pr_trans, pr_emit, pr_prior):
    sys.stderr.write('calculating log likelyhood...\n')
    llh = 0
    for f_sentence, e_sentence in tqdm(bitext):
        llh += (viterbi_decode(f_sentence, e_sentence, pr_trans, pr_emit, pr_prior)[1])
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
    argparser.add_argument("-n", "--num_sentences", dest="num_sents", default=1300, type=int, help="Number of sentences to use for training and alignment")
    argparser.add_argument("--epsilon", dest="epsilon", default=1, type=float, help="Convergence check passes if |L(t_k)-L(t_k-1)|<epsilon")
    argparser.add_argument("--max-iteration", dest="max_iteration", default=100, type=int, help="max number of iteration")
    # argparser.add_argument('--iter', type=int, default=100)
    argparser.add_argument("--save-model", dest="save_model", default="hmmmodel.pickle", help="save variable t")
    argparser.add_argument("--load-model", dest="load_model", help="model file of variable t")
    argparser.add_argument('-r', '--resume', help='resume training')
    argparser.add_argument('--ckpt', default='hmmckpt.pickle', help='check point')
    args = argparser.parse_args()
    f_data = "%s.%s" % (os.path.join(args.datadir, args.fileprefix), args.french)
    e_data = "%s.%s" % (os.path.join(args.datadir, args.fileprefix), args.english)

    if args.logfile:
            logging.basicConfig(filename=args.logfile, filemode='w', level=logging.INFO)

    bitext = [[sentence.strip().split() for sentence in pair] for pair in islice(
        zip(open(f_data), open(e_data)), 0, args.num_sents)]

    if args.load_model:
        iter, pr_trans, pr_emit, pr_prior = load_model(args.load_model)
    else:
        if args.resume:
            iter, pr_trans, pr_emit, pr_prior = load_model(args.resume)
            iter, pr_trans, pr_emit, pr_prior = \
                train(iter, pr_trans, pr_emit, pr_prior, bitext, args.max_iteration, args.ckpt, args.epsilon)
        else:
            f_vocab, e_vocab = build_vocab(bitext)
            print(len(f_vocab), len(e_vocab))
            pr_trans, pr_emit, pr_prior = init_params(bitext, len(f_vocab))
            iter, pr_trans, pr_emit, pr_prior = \
                train(0, pr_trans, pr_emit, pr_prior, bitext, args.max_iteration, args.ckpt, args.epsilon)
        dump_model(args.save_model, iter, pr_trans, pr_emit, pr_prior)
    for f_sentence, e_sentence in bitext:
        J = len(f_sentence)
        alignments, _ = viterbi_decode(f_sentence, e_sentence, pr_trans, pr_emit, pr_prior)
        for j in range(J):
            print("{0}-{1}".format(j, alignments[j]), end=" ")
        print()



if __name__ == '__main__':
    main()