from collections import defaultdict, Counter
import collections
import pprint
import math
import bz2
import string
import argparse
from ngram import LM
from copy import deepcopy
from datetime import datetime

pp = pprint.PrettyPrinter(width=45, compact=True)

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

def score(mappings, cipher_text, lm):
    deciphered = [mappings[cipher_letter] if cipher_letter in mappings else '_' for cipher_letter in cipher_text]
    deciphered = ''.join(deciphered)
    bit_string = [ 'o' if c in mappings else '.' for c in cipher_text]
    bit_string = ''.join(bit_string)

    return lm.score_bitstring(deciphered, bit_string)


def prune(beams, beamsize):
    sorted_beams = sorted(beams, key=lambda b: b[1], reverse=True)

    return sorted_beams[:beamsize]


def beam_search(cipher_text, lm, ext_order, ext_limits, beamsize):
    Hs = []
    Ht = []
    cardinality = 0
    Hs.append(({}, 0))
    Ve = string.ascii_lowercase

    while cardinality < len(ext_order):
        print("Searching for {}/{} letter".format(cardinality, len(ext_order)))
        print("Current size of searching tree: {}".format(len(Hs)))
        cipher_letter = ext_order[cardinality]
        for mappings, sc in Hs:
            for plain_letter in Ve:
                ext_mappings = deepcopy(mappings)
                ext_mappings[cipher_letter] = plain_letter
                if check_limits(ext_mappings, ext_limits, plain_letter):  # only check new added one
                    Ht.append((ext_mappings, score(ext_mappings, cipher_text, lm)))
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
            if count == 6:
                ngrams[count] += 1
            else:
                count += 1
        else:
            ngrams[count] += 1
            count = 0
    if count != 0:
        ngrams[count] += 1
    weights = [0, 0.1, 0.4, 0.6, 1, 1.5, 2]
    score = 0
    for k, v in ngrams.items():
        score += weights[k] * v
    return score

def prune_orders(orders, beamsize):
    sorted_order = sorted(orders, reverse=True)

    return sorted_order[: beamsize]

def search_ext_order(cipher, beamsize):
    symbols = set(cipher)
    freq = Counter(cipher)
    start = ''
    maxf = 0
    for symbol, f in freq.items():
        if f > maxf:
            maxf = f
            start = symbol
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
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-f', '--file', default='data/cipher.txt', help='cipher file')
    arg_parser.add_argument('-b', '--beamsize', type=int, default=100)
    args = arg_parser.parse_args()

    cipher = read_file(args.file)
    cipher = [x for x in cipher if not x.isspace()]
    cipher = ''.join(cipher)
    ext_order = search_ext_order(cipher, 100)
    ext_limits = 8

    freq = Counter(cipher)
    sort_freq = [ kv[0] for kv in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)]

    print(ext_order)
    print(sort_freq)

    print('Loading language model')
    lm = LM("data/6-gram-wiki-char.lm.bz2", n=6, verbose=False)
    print('Language model loaded')

    print('Start deciphering...')
    search_start = datetime.now()
    mappings, sc = beam_search(cipher, lm, ext_order, ext_limits, args.beamsize)
    search_end = datetime.now()
    print('Deciphering completed after {}'.format(search_end - search_start))

    deciphered = [mappings[c] if c in mappings else '_' for c in cipher]
    deciphered = ''.join(deciphered)
    print(deciphered, sc)