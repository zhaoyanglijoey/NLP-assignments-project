from collections import defaultdict, Counter
import collections
import pprint
import math
import bz2
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

def check_limits(mappings, ext_limits):
    targets = mappings.values()
    counts = Counter(targets).values()
    if any([count > ext_limits for count in counts]):
        return False
    else:
        return True

def score(mappings, cipher, lm):
    deciphered = [ mappings[c] if c in mappings else '_' for c in cipher ]
    deciphered = ''.join(deciphered)
    bit_string = [ 'o' if c in mappings else '.' for c in cipher ]
    bit_string = ''.join(bit_string)

    return lm.score_bitstring(deciphered, bit_string)

def prune(beams, beamsize):
    sorted_beams = sorted(beams, key=lambda b: b[1], reverse=True)

    return sorted_beams[:beamsize]

def beam_search(cipher, lm, ext_order, ext_limits, beamsize):
    Hs = []
    Ht = []
    cardinality = 0
    Hs.append(({},0) )
    Ve = 'abcdefghijklmnopqrstuvwxyz'

    while cardinality < len(ext_order):
        f = ext_order[cardinality]
        for mappings, sc in Hs:
            for e in Ve:
                ext_mappings = deepcopy(mappings)
                ext_mappings[f] = e
                if check_limits(ext_mappings, ext_limits):
                    Ht.append((ext_mappings, score(ext_mappings, cipher, lm)))
        Hs = prune(Ht, beamsize)
        cardinality += 1
        Ht = []
        # pp.pprint(Hs)
    Hs.sort(key=lambda b: b[1], reverse=True)
    # pp.pprint(Hs)
    return Hs[0]


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-f', '--file', default='data/cipher.txt', help='cipher file')
    arg_parser.add_argument('-b', '--beamsize', type=int, default=20)
    args = arg_parser.parse_args()

    cipher = read_file(args.file)
    cipher = [x for x in cipher if not x.isspace()]
    cipher = ''.join(cipher)
    freq = Counter(cipher)
    ext_order = [ kv[0] for kv in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)]
    ext_limits = 10

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