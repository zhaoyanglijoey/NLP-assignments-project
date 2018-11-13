import sys
import pickle
import math
from collections import defaultdict
from tqdm import tqdm


def calculate_llh(bitext, t):
    llh = 0
    for f, e in bitext:
        for f_word in f:
            t_sum = 0
            for e_word in e:
                t_sum += t[(f_word, e_word)]
            llh += math.log(t_sum)
    return llh


def train(bitext, f_vocab, e_vocab, max_iteration, epsilon):
    sys.stderr.write("Training...\n")
    t0 = 1/len(f_vocab)
    t = defaultdict(float)

    for f_word in f_vocab:
        for e_word in e_vocab:
            t[(f_word, e_word)] = t0

    llh_old = calculate_llh(bitext, t)
    k = 0
    while True:
        k += 1
        if k > max_iteration:
            sys.stderr.write("Training finished.\n")
            break

        sys.stderr.write("Iteration {0}\n".format(k))
        # Training
        count_pair = defaultdict(float)
        count_e = defaultdict(float)
        for f, e in tqdm(bitext):
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

        with open("ibm_modeL_checkpoint.pickle", 'wb') as f:
            pickle.dump(t, f)

        # Calculate log likelihood
        llh = calculate_llh(bitext, t)
        sys.stderr.write("Log likelihood after iteration {0}: {1}\n".format(k, llh))

        if abs(llh - llh_old) < epsilon:
            sys.stderr.write("Training finished.\n")
            break

        llh_old = llh

    return t
