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

    # for f_word in f_vocab:
    #     for e_word in e_vocab:
    #         t[(f_word, e_word)] = t0
    for f_sentence, e_sentence in bitext:
        for f in f_sentence:
            for e in e_sentence:
                t[(f, e)] = 1/len(f_vocab)

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


def decode(bitext, t):
    alignments_list = []
    sys.stderr.write("Decoding\n")
    for f, e in tqdm(bitext):
        alignments = []
        for i, f_word in enumerate(f):
            best_p = 0
            best_j = 0
            for j, e_word in enumerate(e):
                if t[(f_word, e_word)] > best_p:
                    best_p = t[(f_word, e_word)]
                    best_j = j
            alignments.append((i, best_j))
            # print("{0}-{1}".format(i, best_j), end=" ")
        alignments_list.append(alignments)
        # print()
    return alignments_list


def alignments_intersection(alignments_list_1, alignments_list_2):
    def reverse_alignments(alignments):
        reversed_alignments = []
        for i, j in alignments:
            reversed_alignments.append((j, i))
        return reversed_alignments

    def intersect_lists(list_1, list_2):
        set_2 = set(list_2)
        return [element for element in list_1 if element in set_2]

    alignments_list = []
    for alignments_1, alignments_2 in zip(alignments_list_1, alignments_list_2):
        # alignments_1: alignment from French to English
        # alignments_2: alignment from English to French
        # So we need to make the direction same before intersection
        reverse_alignments_2 = reverse_alignments(alignments_2)
        alignments_list.append(intersect_lists(alignments_1, reverse_alignments_2))
    return alignments_list
