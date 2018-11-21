import matplotlib
matplotlib.use('Agg')
import sys, os
from itertools import islice
import pickle
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import math
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool, Process, Lock, Manager
from multiprocessing.managers import BaseManager, DictProxy
from threading import Thread
from datetime import datetime

class MyManager(BaseManager):
    pass

MyManager.register('defaultdict', defaultdict, DictProxy)

clip = lambda x, l, u: l if x < l else u if x > u else x
def epsilon():
    return 1e-100

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
        # alignment = set([tuple(map(int, x.split("-"))) for x in a.strip().split()])
        alignment = set(a)
        size_a += len(alignment)
        size_s += len(sure)
        size_a_and_s += len(alignment & sure)
        size_a_and_p += len(alignment & possible) + len(alignment & sure)
        if (i < num_display):
            sys.stderr.write("  Alignment %i  KEY: ( ) = guessed, * = sure, ? = possible\n" % i)
            sys.stderr.write("  ")
            for j in ewords:
                sys.stderr.write("---")
            sys.stderr.write("\n")
            for (i, f_i) in enumerate(fwords):
                sys.stderr.write(" |")
                for (j, _) in enumerate(ewords):
                    (left, right) = ("(", ")") if (i, j) in alignment else (" ", " ")
                    point = "*" if (i, j) in sure else "?" if (i, j) in possible else " "
                    sys.stderr.write("%s%s%s" % (left, point, right))
                sys.stderr.write(" | %s\n" % f_i)
            sys.stderr.write("  ")
            for j in ewords:
                sys.stderr.write("---")
            sys.stderr.write("\n")
            for k in range(max(map(len, ewords))):
                sys.stderr.write("  ")
                for word in ewords:
                    letter = word[k] if len(word) > k else " "
                    sys.stderr.write(" %s " % letter)
                sys.stderr.write("\n")
            sys.stderr.write("\n")

    precision = size_a_and_p / size_a
    recall = size_a_and_s / size_s
    aer = 1 - ((size_a_and_s + size_a_and_p) / (size_a + size_s))
    sys.stderr.write("Precision = %f\nRecall = %f\nAER = %f\n" % (precision, recall, aer))
    return precision, recall, aer

class HMMmodel():
    def __init__(self):
        self.pr_trans = {}
        self.pr_emit = {}
        self.pr_prior = {}
        self.pr_word_trans = {}
        self.iter = None

    def init_params(self, bitext):
        sys.stderr.write('initializing parameters...\n')
        self.iter = 0

        # maxe_len = 0
        for f_sentence, e_sentence in bitext:
            I = len(e_sentence)
            # maxe_len = max(maxe_len, I)
            for j, f in enumerate(f_sentence):
                for i, e in enumerate(e_sentence):
                    self.pr_emit[(f, e)] = 1
                # self.pr_emit[(f, 'null')] = 1
            # self.pr_prior[(I, I)] = 0.2
            for i in range(I):
                self.pr_prior[(i, I)] = (1 / I)
                # self.pr_trans[(i+I, i, I)] = 0.2
                # self.pr_word_trans[(i+I, i, e_sentence[i_p], I)] = 0.2
                for i_p in range(I):
                    self.pr_trans[(i, i_p, I)] = (1 / I)
                    self.pr_word_trans[(i, i_p, e_sentence[i_p], I)] = (1 / I)
        # for i in range(maxe_len):
        #     pr_prior[i] = 1 / (maxe_len+1)
        sys.stderr.write('done\n')



    def expectation_step(self, f_sentence, e_sentence, c_emit, c_trans, c_emit_margin, c_prior,
                       c_prior_margin, c_word_trans, c_word_trans_margin, c_stay, c_stay_margin):


        I = len(e_sentence)
        J = len(f_sentence)
        # for i in range(I):
        #     e_sentence.append('null')
        forward_pr, backward_pr = self.forward_backward(f_sentence, e_sentence)
        denominator = 0
        for i in range(I):
            denominator += forward_pr[i][J - 1]
        if denominator == 0:
            # sys.stderr.write('0 denominator!\n')
            return
        # print(denominator)
        for i in range(I):
            for j in range(J):
                gamma = forward_pr[i][j] * backward_pr[i][j] / denominator
                c_emit[(f_sentence[j], e_sentence[i])] += gamma
                c_emit_margin[e_sentence[i]] += gamma
                if j == 0:
                    c_prior[(i, I)] += gamma
                    c_prior_margin[I] += gamma

        for j in range(J - 1):
            for i in range(I):
                for i_p in range(I):
                    si = forward_pr[i_p][j] * self.pr_trans[(i, i_p, I)] * backward_pr[i][j + 1] * \
                         self.pr_emit[(f_sentence[j + 1], e_sentence[i])] / denominator
                    d = clip(i - i_p, -7, 7)
                    c_trans[(d, I)] += si
                    c_word_trans[(d, e_sentence[i_p], I)] += si
                    if i == i_p:
                        c_stay[e_sentence[i_p]] += si
                    c_stay_margin[e_sentence[i_p]] += si

    def get_c_trans(self, bitext, forward_and_backward_prs):
        c_trans = defaultdict(float)
        for i, (f_sentence, e_sentence) in enumerate(tqdm(bitext)):
            I = len(e_sentence)
            J = len(f_sentence)

            forward_pr, backward_pr, _, sis = forward_and_backward_prs[i]

            if sis is None:
                continue

            for j in range(J - 1):
                for i in range(I):
                    for i_p in range(I):
                        si = sis[i][i_p][j]
                        d = clip(i - i_p, -7, 7)
                        c_trans[(d, I)] += si

        return c_trans

    def get_c_word_trans(self, bitext, forward_and_backward_prs):
        c_word_trans = defaultdict(float)
        for i, (f_sentence, e_sentence) in enumerate(tqdm(bitext)):
            I = len(e_sentence)
            J = len(f_sentence)

            forward_pr, backward_pr, _, sis = forward_and_backward_prs[i]

            if sis is None:
                continue
            for j in range(J - 1):
                for i in range(I):
                    for i_p in range(I):
                        si = sis[i][i_p][j]
                        d = clip(i - i_p, -7, 7)
                        c_word_trans[(d, e_sentence[i_p], I)] += si

        return c_word_trans

    def get_c_stay(self, bitext, forward_and_backward_prs):
        c_stay = defaultdict(float)
        c_stay_margin = defaultdict(epsilon)
        for i, (f_sentence, e_sentence) in enumerate(tqdm(bitext)):
            I = len(e_sentence)
            J = len(f_sentence)

            forward_pr, backward_pr, _, sis = forward_and_backward_prs[i]

            if sis is None:
                continue
            for j in range(J - 1):
                for i in range(I):
                    for i_p in range(I):
                        si = sis[i][i_p][j]
                        if i == i_p:
                            c_stay[e_sentence[i_p]] += si
                        c_stay_margin[e_sentence[i_p]] += si

        return c_stay, c_stay_margin

    def get_c_emit_prior(self, bitext, forward_and_backward_prs):
        c_emit = defaultdict(float)
        c_emit_margin = defaultdict(epsilon)
        c_prior = defaultdict(float)
        c_prior_margin = defaultdict(epsilon)

        for i, (f_sentence, e_sentence) in enumerate(tqdm(bitext)):
            I = len(e_sentence)
            J = len(f_sentence)

            forward_pr, backword_pr, gammas, _ = forward_and_backward_prs[i]

            if gammas is None:
                continue
            for i in range(I):
                for j in range(J):
                    gamma = gammas[i][j]
                    c_emit[(f_sentence[j], e_sentence[i])] += gamma
                    c_emit_margin[e_sentence[i]] += gamma
                    if j == 0:
                        c_prior[(i, I)] += gamma
                        c_prior_margin[I] += gamma

        return c_emit, c_emit_margin, c_prior, c_prior_margin


    def train(self, bitext, max_iteration, ckpt,
                f_data, e_data, a_data, validate = False):

        e_lens = set()

        for f_sentence, e_sentence in bitext:
            e_lens.add(len(e_sentence))
        f_vocab, e_vocab = build_vocab(bitext)
        f_vocab_size = len(f_vocab)
        # if not no_break:
        #     prev_llh = self.validate(bitext, f_data, e_data, a_data)[0]
        #     sys.stderr.write('iteration {}, llh {}\n'.format(self.iter, prev_llh))

        self.aers = {}

        clip = lambda x, l, u: l if x < l else u if x > u else x

        p_null = 0.2
        alpha = 0.4
        beta = 0.0
        lambd = 0.1
        tau = 1000
        local_iter = 0
        # pool = Pool()
        # mgr = MyManager()
        # mgr.start()

        while local_iter < max_iteration:
            self.iter += 1
            local_iter += 1
            sys.stderr.write('training iter {}...\n'.format(self.iter))
            c_emit = defaultdict(float)
            c_emit_margin = defaultdict(epsilon)
            c_prior = defaultdict(float)
            c_prior_margin = defaultdict(epsilon)
            c_trans = defaultdict(float)
            c_word_trans = defaultdict(float)
            c_stay = defaultdict(float)
            c_stay_margin = defaultdict(epsilon)

            # c_emit = mgr.defaultdict(float)
            # c_trans = mgr.defaultdict(float)
            # c_emit_margin = mgr.defaultdict(epsilon)
            # c_prior = mgr.defaultdict(float)
            # c_prior_margin = mgr.defaultdict(epsilon)
            # c_word_trans = mgr.defaultdict(float)
            # c_word_trans_margin = mgr.defaultdict(epsilon)
            # c_stay = mgr.defaultdict(float)
            # c_stay_margin = mgr.defaultdict(epsilon)
            # start = datetime.now()
            # forward_and_backward_prs = pool.map(self.forward_backward, bitext)
            #
            # c_emit_prior_res = \
            #     pool.apply_async(self.get_c_emit_prior, args=(bitext, forward_and_backward_prs))
            #
            # c_trans_res = pool.apply_async(self.get_c_trans, args=(bitext, forward_and_backward_prs))
            # c_word_trans_res = pool.apply_async(self.get_c_word_trans, args=(bitext, forward_and_backward_prs))
            # c_stay_res = pool.apply_async(self.get_c_stay, args=(bitext, forward_and_backward_prs))
            #
            # c_emit, c_emit_margin, c_prior, c_prior_margin = c_emit_prior_res.get()
            # c_trans = c_trans_res.get()
            # c_word_trans = c_word_trans_res.get()
            # c_stay, c_stay_margin = c_stay_res.get()

            for f_sentence, e_sentence in tqdm(bitext):
                I = len(e_sentence)
                J = len(f_sentence)
                # for i in range(I):
                #     e_sentence.append('null')
                forward_pr, backword_pr = self.forward_backward([f_sentence, e_sentence])

                denominator = 0
                for i in range(I):
                    denominator += forward_pr[i][J-1]
                if denominator == 0:
                    # sys.stderr.write('0 denominator!\n')
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
                            si = forward_pr[i_p][j] * self.pr_trans[(i, i_p, I)] * backword_pr[i][j+1] * \
                                 self.pr_emit[(f_sentence[j+1], e_sentence[i])] / denominator
                            d = clip(i - i_p, -7, 7)
                            c_trans[(d, I)] += si
                            c_word_trans[(d, e_sentence[i_p])] += si
                            if i == i_p:
                                c_stay[e_sentence[i_p]] += si
                            c_stay_margin[e_sentence[i_p]] += si
            # print(datetime.now() - start)

            for f_sentence, e_sentence in bitext:
                for f in f_sentence:
                    for e in e_sentence:
                        self.pr_emit[(f, e)] = (beta * (1 / f_vocab_size) + (1-beta) * (c_emit[(f, e)] / c_emit_margin[e]))
            for I in e_lens:
                for i_p in range(I):
                    margin = 0
                    c_m7 = 0
                    c_p7 = 0
                    for i_pp in range(I):
                        if i_pp - i_p < -7:
                            c_m7 += 1
                        elif i_pp - i_p == -7:
                            c_m7 += 1
                            margin += c_trans[(-7, I)]
                        elif i_pp - i_p > 7:
                            c_p7 += 1
                        elif i_pp - i_p == 7:
                            c_p7 += 1
                            margin += c_trans[(7, I)]
                        else:
                            margin += c_trans[(i_pp-i_p, I)]
                    if margin == 0:
                        # sys.stderr.write('0 transition count!\n')
                        continue
                    for i in range(I):
                        if i - i_p <= -7:
                            self.pr_trans[(i, i_p, I)] = alpha * 1 / I + (1 - alpha) * (c_trans[(-7, I)] / c_m7 / margin)
                        elif i - i_p >= 7:
                            self.pr_trans[(i, i_p, I)] = alpha * 1 / I + (1 - alpha) * (c_trans[(7, I)] / c_p7 / margin)
                        else:
                            self.pr_trans[(i, i_p, I)] = alpha * 1 / I + (1-alpha) * (c_trans[(i-i_p, I)] / margin)
                for i in range(I):
                    self.pr_prior[(i, I)] = alpha * 1 / I + (1-alpha) * (c_prior[(i, I)] / c_prior_margin[I])

            for _, e_sentence in bitext:
                I = len(e_sentence)
                for i_p in range(I):
                    word_trans_margin_i_p = 0
                    c_m7 = 0
                    c_p7 = 0
                    for i_pp in range(I):
                        if i_pp - i_p < -7:
                            c_m7 += 1
                        elif i_pp - i_p == -7:
                            c_m7 += 1
                            word_trans_margin_i_p += c_word_trans[(i_pp - i_p, e_sentence[i_p])]
                        elif i_pp - i_p > 7:
                            c_p7 += 1
                        elif i_pp - i_p == 7:
                            c_p7 += 1
                            word_trans_margin_i_p += c_word_trans[(i_pp - i_p, e_sentence[i_p])]
                        else:
                            word_trans_margin_i_p += c_word_trans[(i_pp - i_p, e_sentence[i_p])]

                    for i in range(I):
                        if i == i_p:
                            p_stay = c_stay[e_sentence[i_p]] /  c_stay_margin[e_sentence[i_p]]
                            self.pr_word_trans[(i, i_p, e_sentence[i_p], I)] = \
                                lambd * self.pr_trans[(i, i_p, I)] + (1-lambd) * p_stay
                        else:
                            d = clip(i - i_p, -7, 7)
                            if i - i_p <= -7:
                                self.pr_word_trans[(i, i_p, e_sentence[i_p], I)] = \
                                (c_word_trans[(d, e_sentence[i_p])] / c_m7 + tau * self.pr_trans[(i, i_p, I)]) \
                                / (word_trans_margin_i_p + tau)
                            elif i - i_p >= 7:
                                self.pr_word_trans[(i, i_p, e_sentence[i_p], I)] = \
                                (c_word_trans[(d, e_sentence[i_p])] / c_p7 + tau * self.pr_trans[(i, i_p, I)]) \
                                / (word_trans_margin_i_p + tau)
                            else:
                                self.pr_word_trans[(i, i_p, e_sentence[i_p], I)] = \
                                    (c_word_trans[(d, e_sentence[i_p])] + tau * self.pr_trans[(i, i_p, I)]) \
                                    / (word_trans_margin_i_p + tau)

            if ckpt is not None:
                self.dump_model(os.path.join(ckpt, 'iter{:03}.hmm'.format(self.iter)))

            if validate:
                llh, aer = self.validate(bitext, f_data, e_data, a_data)
                # sys.stderr.write('iteration {}, llh {}\n'.format(self.iter, llh))
                self.aers[self.iter] = aer
                # if abs(llh - prev_llh) < epsilon:
                #     break
                # prev_llh = llh

        # if validate:
        #     plt.figure()
        #     tmp = sorted(aers.items())
        #     plt.plot([item[0] for item in tmp], [item[1] for item in tmp], '-')
        #     plt.xlabel('iteration')
        #     plt.ylabel('AER')
        #     plt.savefig('AER.png')
        return self

    def forward_backward(self, pair):
        f_sentence, e_sentence = pair

        I = len(e_sentence)
        J = len(f_sentence)

        # forward_pr = np.zeros((I, J), dtype=np.float64)
        # backward_pr = np.zeros((I, J), dtype=np.float64)
        forward_pr = [[0.] * J for _ in range(I)]
        backward_pr = [[0.] * J for _ in range(I)]

        for i in range(I):
            forward_pr[i][0] = self.pr_prior[(i, I)] * self.pr_emit[(f_sentence[0], e_sentence[i])]

        for j in range(1, J):
            for i in range(I):
                trans = 0
                for i_p in range(I):
                    trans += forward_pr[i_p][j - 1] * self.pr_word_trans[(i, i_p, e_sentence[i_p], I)]
                forward_pr[i][j] = self.pr_emit[(f_sentence[j], e_sentence[i])] * trans

        for i in range(I):
            backward_pr[i][J - 1] = 1

        for j in range(J - 1)[::-1]:
            for i_p in range(I):
                tmp = 0
                for i in range(I):
                    tmp += backward_pr[i][j + 1] * self.pr_word_trans[(i, i_p, e_sentence[i_p], I)] * \
                           self.pr_emit[(f_sentence[j + 1], e_sentence[i])]
                backward_pr[i_p][j] = tmp

        # denominator = 0
        # for i in range(I):
        #     denominator += forward_pr[i][J - 1]
        # if denominator == 0:
        #     return (forward_pr, backward_pr, None, None)
        #
        # gamma = [[0.] * J for _ in range(I)]
        # for i in range(I):
        #     for j in range(J):
        #         gamma[i][j] = forward_pr[i][j] * backward_pr[i][j] / denominator
        #
        # si = [[[0.] * J for _ in range(I)] for _ in range(I)]
        # for j in range(J - 1):
        #     for i in range(I):
        #         for i_p in range(I):
        #             si[i][i_p][j] = forward_pr[i_p][j] * self.pr_trans[(i, i_p, I)] * backward_pr[i][j + 1] * \
        #                  self.pr_emit[(f_sentence[j + 1], e_sentence[i])] / denominator
        #
        # return (forward_pr, backward_pr, gamma, si)
        return forward_pr, backward_pr

    def viterbi_decode(self, f_sentence, e_sentence):
        I = len(e_sentence)
        J = len(f_sentence)
        V = [[0.] * J for _ in range(I)]
        # V = np.zeros((I, J), dtype=float)
        backptr = [[0] * J for _ in range(I)]
        for i in range(I):
            try:
                V[i][0] = math.log(self.pr_prior[(i, I)]) + math.log(self.pr_emit[(f_sentence[0], e_sentence[i])])
            except ValueError:
                V[i][0] = -10000000

            # V[i][0] = pr_prior[(i, I)] * pr_emit[(f_sentence[0], e_sentence[i])]

        for j in range(1, J):
            for i in range(I):
                for i_p in range(I):
                    try:
                        tmp = V[i_p][j - 1] + math.log(self.pr_word_trans[(i, i_p, e_sentence[i_p], I)]) + \
                              math.log(self.pr_emit[(f_sentence[j], e_sentence[i])])
                    except ValueError:
                        tmp = -10000000
                    # tmp = V[i_p][j-1] * pr_trans[(i, i_p, I)] * pr_emit[(f_sentence[j], e_sentence[i])]
                    if i_p == 0 or (i_p != 0 and tmp > V[i][j]):
                        V[i][j] = tmp
                        backptr[i][j] = i_p
        best_idx = 0
        best_score = V[best_idx][J - 1]
        for i in range(1, I):
            if V[i][J - 1] > best_score:
                best_score = V[i][J - 1]
                best_idx = i

        alignments = [best_idx]
        for j in range(1, J)[::-1]:
            best_idx = backptr[best_idx][j]
            alignments.append(best_idx)

        alignments.reverse()
        return (alignments, best_score)

    def decode(self, bitext):
        # sys.stderr.write('calculating log likelyhood...\n')
        llh = 0
        output = []
        for f_sentence, e_sentence in (bitext[:447]):
            alignments, llh_t = self.viterbi_decode(f_sentence, e_sentence)
            llh += llh_t
            output_line = []
            for j in range(len(f_sentence)):
                # output_line += "{0}-{1} ".format(j, alignments[j])
                output_line.append((j, alignments[j]))
            output.append(output_line)

        return output, llh

    def validate(self, bitext, f_data, e_data, a_data):
        alignments_list, llh = self.decode(bitext)
        trizip = zip(f_data, e_data, a_data, alignments_list)
        aer = score_alignments(trizip)[2]

        return llh, aer

    def dump_model(self, file):
        model = self.get_model_params()
        with open(file, 'wb') as f:
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

    def get_model_params(self):
        model = {
            'iter': self.iter,
            'pr_trans': self.pr_trans,
            'pr_emit': self.pr_emit,
            'pr_prior': self.pr_prior,
            'pr_word_trans': self.pr_word_trans
        }

        return model

    def load_model(self, file):
        with open(file, 'rb') as f:
            model = pickle.load(f)
        self.set_model_params(model)

    def set_model_params(self, model):
        self.iter = model['iter']
        self.pr_trans = model['pr_trans']
        self.pr_emit = model['pr_emit']
        self.pr_prior = model['pr_prior']
        self.pr_word_trans = model['pr_word_trans']

    def load_from_ibm1(self, ibm1model):
        self.pr_emit = ibm1model

class BiHMMmodel():
    def __init__(self):
        self.forward_model = HMMmodel()
        self.backward_model = HMMmodel()

    def init_params(self, bitext, rev_bitext):
        sys.stderr.write('initializing parameters...\n')
        self.forward_model.init_params(bitext)
        self.backward_model.init_params(rev_bitext)

    def load_forward(self, file):
        self.forward_model.load_model(file)

    def load_backward(self, file):
        self.backward_model.load_model(file)

    def dump_model(self, file):
        model = {
            'forward': self.forward_model.get_model_params(),
            'backward': self.backward_model.get_model_params()
        }
        with open(file, 'wb') as f:
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

    def load_model(self, file):
        with open(file, 'rb') as f:
            model = pickle.load(f)

        self.forward_model.set_model_params(model['forward'])
        self.backward_model.set_model_params(model['backward'])

    def load_from_ibm1(self, file):
        with open(file, 'rb') as f:
            model = pickle.load(f)
        self.forward_model.load_from_ibm1(model[0])
        self.backward_model.load_from_ibm1(model[1])

    def reverse_alignments_list(self, alignments_list):
        reversed_alignments_list = []
        for alignments in alignments_list:
            reversed_alignments = []
            for i, j in alignments:
                reversed_alignments.append((j, i))
            reversed_alignments_list.append(reversed_alignments)

        return reversed_alignments_list

    def intersect_lists(self, list_1, list_2):
        set_2 = set(list_2)
        return [element for element in list_1 if element in set_2]

    def decode(self, bitext, rev_bitext):
        alignment_forward, _ = self.forward_model.decode(bitext)
        alignment_backward, _ = self.backward_model.decode(rev_bitext)
        alignment_backward_rev = self.reverse_alignments_list(alignment_backward)

        alignments_list = []

        for af, ab in zip(alignment_forward, alignment_backward_rev):
            alignments_list.append(self.intersect_lists(af, ab))

        return alignments_list

    def validate(self, bitext, rev_bitext, f_data, e_data, a_data):
        # self.forward_model.validate(bitext, f_data, e_data, a_data)
        alignments_list = self.decode(bitext, rev_bitext)
        aer = score_alignments(zip(f_data, e_data, a_data, alignments_list))[2]

    def train(self, bitext, rev_bitext, max_iteration, ckpt,
                f_data, e_data, a_data, validate = True):

        aer_bi = {}
        pool = Pool(1)
        for i in range(max_iteration):


            sys.stderr.write('training forward model...\n')
            forward_res = pool.apply_async(func=self.forward_model.train, args=(
                bitext, 1, None, f_data, e_data, a_data, validate))
            # t_forward = Thread(target=self.forward_model.train, args=(
            #     bitext, 1, None, f_data, e_data, a_data, validate))
            # t_forward.start()

            # p_forward = multiprocessing.Process(target=self.forward_model.train, args=(
            #     bitext, 1, None, f_data, e_data, a_data, validate))
            # p_forward.start()
            # self.forward_model.train(bitext, 1, None, f_data, e_data, a_data, validate = validate)

            sys.stderr.write('training backward model...\n')
            self.backward_model.train(rev_bitext, 1, None, f_data, e_data, a_data, validate = False)
            # backward_res = pool.apply_async(func=self.backward_model.train, args=(
            #     rev_bitext, 1, None, f_data, e_data, a_data, False))
            # t_backward = Thread(target=self.backward_model.train, args=(
            #     rev_bitext, 1, None, f_data, e_data, a_data, False))
            # t_backward.start()
            # p_backward = multiprocessing.Process(target=self.backward_model.train, args=(
            #     rev_bitext, 1, None, f_data, e_data, a_data, False))
            # p_backward.start()
            # self.backward_model.train(rev_bitext, 1, None, f_data, e_data, a_data, validate = False)

            self.forward_model = forward_res.get()
            # self.backward_model = backward_res.get()
            # pool.close()
            # pool.join()

            # t_forward.join()
            # t_backward.join()
            # p_forward.join()
            # p_backward.join()

            if ckpt is not None:
                self.dump_model(os.path.join(ckpt, 'bihmm_iter{}.m'.format(self.forward_model.iter)))

            if validate:
                aer_bi[i+1] = self.validate(bitext, rev_bitext, f_data, e_data, a_data)

        if validate:
            plt.figure()
            iters = sorted(aer_bi.keys())
            plt.plot(iters, [aer_bi[i] for i in iters], '-')
            plt.plot(iters, [self.forward_model.aers[i] for i in iters], '-')
            plt.legend(['bidirection_hmm', 'hmm'])
            plt.xlabel('iteration')
            plt.ylabel('AER')
            plt.savefig('AER_bi.png')
