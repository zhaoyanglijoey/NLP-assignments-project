import argparse, sys, os, logging
from itertools import islice
import pickle
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import math
import matplotlib.pyplot as plt

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
    f_list = []
    e_list = []
    for f, e in bitext:
        f_list += f
        e_list += e
    f_vocab = set(f_list)
    e_vocab = set(e_list)
    return (f_vocab, e_vocab)

def score_alignments(trizip, num_display = 0):
    (size_a, size_s, size_a_and_s, size_a_and_p) = (0.0, 0.0, 0.0, 0.0)

    for (i, (f, e, g, a)) in enumerate(trizip):
        fwords = f.strip().split()
        ewords = e.strip().split()
        sure = set([tuple(map(int, x.split("-"))) for x in filter(lambda x: x.find("-") > -1, g.strip().split())])
        possible = set([tuple(map(int, x.split("?"))) for x in filter(lambda x: x.find("?") > -1, g.strip().split())])
        alignment = set([tuple(map(int, x.split("-"))) for x in a.strip().split()])
        size_a += len(alignment)
        size_s += len(sure)
        size_a_and_s += len(alignment & sure)
        size_a_and_p += len(alignment & possible) + len(alignment & sure)
        if (i < num_display):
            sys.stdout.write("  Alignment %i  KEY: ( ) = guessed, * = sure, ? = possible\n" % i)
            sys.stdout.write("  ")
            for j in ewords:
                sys.stdout.write("---")
            sys.stdout.write("\n")
            for (i, f_i) in enumerate(fwords):
                sys.stdout.write(" |")
                for (j, _) in enumerate(ewords):
                    (left, right) = ("(", ")") if (i, j) in alignment else (" ", " ")
                    point = "*" if (i, j) in sure else "?" if (i, j) in possible else " "
                    sys.stdout.write("%s%s%s" % (left, point, right))
                sys.stdout.write(" | %s\n" % f_i)
            sys.stdout.write("  ")
            for j in ewords:
                sys.stdout.write("---")
            sys.stdout.write("\n")
            for k in range(max(map(len, ewords))):
                sys.stdout.write("  ")
                for word in ewords:
                    letter = word[k] if len(word) > k else " "
                    sys.stdout.write(" %s " % letter)
                sys.stdout.write("\n")
            sys.stdout.write("\n")

    precision = size_a_and_p / size_a
    recall = size_a_and_s / size_s
    aer = 1 - ((size_a_and_s + size_a_and_p) / (size_a + size_s))
    sys.stdout.write("Precision = %f\nRecall = %f\nAER = %f\n" % (precision, recall, aer))
    return precision, recall, aer

def init_params(bitext):
    sys.stderr.write('initializing parameters...\n')

    pr_trans = {}
    pr_emit = {}
    pr_prior = {}

    # maxe_len = 0
    for f_sentence, e_sentence in bitext:
        I = len(e_sentence)
        # maxe_len = max(maxe_len, I)
        for j, f in enumerate(f_sentence):
            for i, e in enumerate(e_sentence):
                pr_emit[(f, e)] = 1
        for i in range(I):
            pr_prior[(i, I)] = 1 / I
            for i_p in range(I):
                pr_trans[(i, i_p, I)] = 1 / I
    # for i in range(maxe_len):
    #     pr_prior[i] = 1 / (maxe_len+1)
    sys.stderr.write('done\n')
    return pr_trans, pr_emit, pr_prior

def forward_backward(f_sentence, e_sentence, pr_trans, pr_emit, pr_prior, scale):
    I = len(e_sentence)
    J = len(f_sentence)

    # forward_pr = np.zeros((I, J), dtype=np.float64)
    # backward_pr = np.zeros((I, J), dtype=np.float64)
    forward_pr = [[0.]*J for _ in range(I)]
    backward_pr = [[0.]*J for _ in range(I)]

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


def train(iter, pr_trans, pr_emit, pr_prior, bitext, max_iteration, ckpt,
            f_data, e_data, a_data, num_sents, epsilon = None, no_break = False):
    e_lens = set()

    for f_sentence, e_sentence in bitext:
        e_lens.add(len(e_sentence))
    f_vocab, e_vocab = build_vocab(bitext)
    f_vocab_size = len(f_vocab)
    if not no_break:
        prev_llh = calc_llh(bitext, pr_trans, pr_emit, pr_prior,
                            f_data, e_data, a_data, num_sents)[0]
        sys.stderr.write('iteration {}, llh {}\n'.format(iter, prev_llh))

    aers = {}

    alpha = 0.1
    beta = 0.0
    while iter < max_iteration:
        iter += 1
        sys.stderr.write('training iter {}...\n'.format(iter))
        c_emit = defaultdict(float)
        c_trans = defaultdict(float)
        c_emit_margin = defaultdict(lambda: 1e-100)
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
                sys.stderr.write('0 denominator!\n')
                continue
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

        dump_model(os.path.join(ckpt, 'iter{:03}.hmm'.format(iter)), iter, pr_trans, pr_emit, pr_prior)

        if not no_break:
            llh, aer = calc_llh(bitext, pr_trans, pr_emit, pr_prior,
                           f_data, e_data, a_data, num_sents)
            sys.stderr.write('iteration {}, llh {}\n'.format(iter, llh))
            aers[iter] = aer
            if abs(llh - prev_llh) < epsilon:
                break
            prev_llh = llh
    if not no_break:
        plt.figure()
        tmp = sorted(aers.items())
        plt.plot([item[0] for item in tmp], [item[1] for item in tmp], '-')
        plt.xlabel('iteration')
        plt.ylabel('AER')
        plt.savefig('AER.png')

    return iter, pr_trans, pr_emit, pr_prior

def viterbi_decode(f_sentence, e_sentence, pr_trans, pr_emit, pr_prior):
    I = len(e_sentence)
    J = len(f_sentence)
    V = [[0.]*J for _ in range(I)]
    # V = np.zeros((I, J), dtype=float)
    backptr = [[0]*J for _ in range(I)]
    for i in range(I):
        try:
            V[i][0] = math.log(pr_prior[(i, I)]) + math.log(pr_emit[(f_sentence[0], e_sentence[i])])
        except ValueError:
            V[i][0] = -10000000

        # V[i][0] = pr_prior[(i, I)] * pr_emit[(f_sentence[0], e_sentence[i])]

    for j in range(1, J):
        for i in range(I):
            for i_p in range(I):
                try:
                    tmp = V[i_p][j-1] + math.log(pr_trans[(i, i_p, I)]) +  \
                              math.log(pr_emit[(f_sentence[j], e_sentence[i])])
                except ValueError:
                    tmp = -10000000
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

def calc_llh(bitext, pr_trans, pr_emit, pr_prior, f_data, e_data, a_data, num_sents):
    sys.stderr.write('calculating log likelyhood...\n')
    llh = 0
    output = []
    for f_sentence, e_sentence in tqdm(bitext):
        alignments, llh_t = viterbi_decode(f_sentence, e_sentence, pr_trans, pr_emit, pr_prior)
        llh += llh_t
        output_line = ''
        for j in range(len(f_sentence)):
            output_line += "{0}-{1} ".format(j, alignments[j])
        output.append(output_line)
    trizip = zip(f_data, e_data, a_data, output)
    aer = score_alignments(trizip)[2]
    # sys.stderr.write('done\n')
    return llh, aer

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
    if args.load_model:
        iter, pr_trans, pr_emit, pr_prior = load_model(args.load_model)
    else:
        if not os.path.exists(args.ckptdir):
            os.mkdir(args.ckptdir)

        if args.resume:
            iter, pr_trans, pr_emit, pr_prior = load_model(args.resume)
            iter, pr_trans, pr_emit, pr_prior = \
                train(iter, pr_trans, pr_emit, pr_prior, bitext, args.max_iteration,
                      args.ckptdir, f_data, e_data, a_data, args.num_sents, args.epsilon)
        else:
            pr_trans, pr_emit, pr_prior = init_params(bitext)
            iter, pr_trans, pr_emit, pr_prior = \
                train(0, pr_trans, pr_emit, pr_prior, bitext, args.max_iteration,
                      args.ckptdir, f_data, e_data, a_data, args.num_sents, args.epsilon)
        dump_model(args.save_model, iter, pr_trans, pr_emit, pr_prior)
    for f_sentence, e_sentence in bitext:
        J = len(f_sentence)
        alignments, _ = viterbi_decode(f_sentence, e_sentence, pr_trans, pr_emit, pr_prior)
        for j in range(J):
            print("{0}-{1}".format(j, alignments[j]), end=" ")
        print()



if __name__ == '__main__':
    main()