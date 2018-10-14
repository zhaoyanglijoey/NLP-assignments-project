from collections import defaultdict, Counter
import collections
import pprint
import math
import bz2
import string
import argparse
from ngram import LM
from nlm_scorer import NlmScorer
import nlm
import evaluator
from copy import deepcopy
from datetime import datetime
# from multiprocessing import Pool
import torch
from torch.multiprocessing import Pool, set_start_method

try:
    torch.multiprocessing.set_start_method('spawn')
except RuntimeError:
    print('Start method already set to spawn!')

pp = pprint.PrettyPrinter(width=45, compact=True)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-f', '--file', default='data/cipher.txt', help='cipher file')
arg_parser.add_argument('-b', '--beamsize', type=int, default=1000)
arg_parser.add_argument('--cuda', action='store_true', default=False)
arg_parser.add_argument('-nw', '--num-workers', type=int, default=12)
args = arg_parser.parse_args()

lm_order = 6
contiguous_score_weights = [0,0,1,1,1,2,3]

ext_limits = 7

print('Loading language model')
lm = LM("data/6-gram-wiki-char.lm.bz2", n=lm_order, verbose=False)
# model = nlm.load_model("data/mlstm_ns.pt", cuda=args.cuda)
# nlm = NlmScorer(model, cuda=args.cuda)
nlm = None
print('Language model loaded')
mem = {}
mem_start = {}

def read_file(filename):
    if filename[-4:] == ".bz2":
        with bz2.open(filename, 'rt') as f:
            content = f.read()
            f.close()
    else:
        with open(filename, 'r') as f:
            content = f.read()
            f.close()
    return content

def check_limits(mappings, ext_limits, letter_to_check=0):
    if letter_to_check is None:
        targets = mappings.values()
        counts = Counter(targets).values()
        if any([count > ext_limits for count in counts]):
            return False
        else:
            return True
    else:
        plaintext_letters = list(mappings.values())
        return plaintext_letters.count(letter_to_check) <= ext_limits

def score_single_seq(t):
    i, seq = t
    # if len(seq) >= 20:
    #     print('Scoring:', seq)
    #     return nlm.score_seq(seq)
    #
    # else:
    #     return lm.score_partial_seq(seq) if i != 0 else lm.score_seq(seq)
    if i == 0:
        if seq not in mem_start:
            mem_start[seq] = lm.score_seq(seq)
        return mem_start[seq]
    else:
        if seq not in mem:
            mem[seq] = lm.score_partial_seq(seq)
        return mem[seq]

pool = Pool(args.num_workers)

def score(mappings, cipher_text, lm, nlm):
    deciphered = [mappings[cipher_letter] if cipher_letter in mappings else ' ' for cipher_letter in cipher_text]
    deciphered = ''.join(deciphered)
    # bit_string = [ 'o' if c in mappings else '.' for c in cipher_text]
    # bit_string = ''.join(bit_string)
    seqs = deciphered.split()

    res = sum(pool.map(score_single_seq, zip(range(len(seqs)),seqs)))

    # return lm.score_bitstring(deciphered, bit_string)
    return res

def prune(beams, beamsize):
    sorted_beams = sorted(beams, key=lambda b: b[1], reverse=True)

    return sorted_beams[:beamsize]


def beam_search(cipher_text, lm, nlm, ext_order, ext_limits, init_beamsize):
    Hs = []
    Ht = []
    cardinality = 0
    Hs.append(({}, 0))
    Ve = string.ascii_lowercase
    scorer = lm

    while cardinality < len(ext_order):
        beamsize = int(init_beamsize*(0.94**cardinality))
        # beamsize = init_beamsize
        # if cardinality > 10:
        #     scorer = nlm
        print("Searching for {}/{} letter".format(cardinality, len(ext_order)))
        print("\tCurrent size of searching tree: {:,}".format(len(Hs)))
        print("\tGoing to be expended to: {:,}".format(len(Hs) * len(Ve)))
        cipher_letter = ext_order[cardinality]
        for mappings, sc in Hs:
            for plain_letter in Ve:
                ext_mappings = deepcopy(mappings)
                ext_mappings[cipher_letter] = plain_letter
                if check_limits(ext_mappings, ext_limits, plain_letter):  # only check new added one
                    Ht.append((ext_mappings, score(ext_mappings, cipher_text, lm, nlm)))
        Hs = prune(Ht, beamsize)
        cardinality += 1
        Ht = []
        # print(Hs)
    Hs.sort(key=lambda b: b[1], reverse=True)
    # pp.pprint(Hs)
    return Hs[0]

def contiguous_score(cipher, order):
    order = set(order)
    count = 0
    ngrams = defaultdict(int)
    for c in cipher:
        if c in order:
            count += 1
            if count >= lm_order:
                ngrams[lm_order] += 1
            else:
                ngrams[count] += 1
        else:
            count = 0
    score = 0
    for k, v in ngrams.items():
        score += contiguous_score_weights[k] * v
    return score

def prune_orders(orders, beamsize):
    sorted_order = sorted(orders, reverse=True)

    return sorted_order[: beamsize]

def search_ext_order(cipher, beamsize):
    symbols = set(cipher)
    # Start with the most common character
    freq = Counter(cipher)
    start = freq.most_common(1)[0][0]
    orders = [([0], [start])]
    orders_tmp = []
    symbols.remove(start)
    for i in range(len(symbols)):
        for scores, order in orders:
            for symbol in symbols:
                if symbol not in order:
                    new_order = deepcopy(order)
                    new_order.append(symbol)
                    new_scores = deepcopy(scores)
                    new_scores.insert(0, contiguous_score(cipher, new_order))
                    orders_tmp.append((new_scores, new_order))
        orders = prune_orders(orders_tmp, beamsize)
        orders_tmp = []
        # pp.pprint(orders)
    orders.sort(reverse=True)
    # pp.pprint(orders)
    return orders[0][1]


if __name__ == '__main__':
    # arg_parser = argparse.ArgumentParser()
    # arg_parser.add_argument('-f', '--file', default='data/cipher.txt', help='cipher file')
    # arg_parser.add_argument('-b', '--beamsize', type=int, default=100)
    # arg_parser.add_argument('--cuda', action='store_true', default=False)
    # args = arg_parser.parse_args()

    cipher = read_file(args.file)
    cipher = [x for x in cipher if not x.isspace()]
    cipher = ''.join(cipher)
    ext_order = search_ext_order(cipher, 100)

    # freq = Counter(cipher)
    # sort_freq = [ kv[0] for kv in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)]

    print(ext_order)
    # print(sort_freq)

    # print('Loading language model')
    # lm = LM("data/6-gram-wiki-char.lm.bz2", n=6, verbose=False)
    # model = nlm.load_model("data/mlstm_ns.pt", cuda=args.cuda)
    # nlm = NlmScorer(model, cuda=args.cuda)
    # print('Language model loaded')

    print('Start deciphering...')
    search_start = datetime.now()
    mappings, sc = beam_search(cipher, lm, nlm, ext_order, ext_limits, args.beamsize)

    search_end = datetime.now()
    print('Deciphering completed after {}'.format(search_end - search_start))
    print(mappings)
    deciphered = [mappings[c] if c in mappings else '_' for c in cipher]
    deciphered = ''.join(deciphered)
    print(deciphered, sc)
    print(evaluator.evaluate(deciphered))