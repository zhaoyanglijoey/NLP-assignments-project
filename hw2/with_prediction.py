# Improvement according to Anoop's paper.
# Predict unknown letters and then calculate score with nlm and frequency matching heuristic.

from collections import defaultdict, Counter
import collections
import pprint
import math
import bz2
import string
import argparse
from ngram import LM
import nlm
from copy import deepcopy
from datetime import datetime
import pdb

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


def check_limits(mappings, ext_limits, letter_to_check=None):
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


def score(Ve, mappings, cipher_text, cipher_letter, old_score, lm_order, lm, model, cuda, cipher_freq, plaintxt_freq):
    # Replace with existing mappings
    deciphered = [mappings[cipher_letter] if cipher_letter in mappings else '_' for cipher_letter in cipher_text]
    score = 0

    print(mappings, end='： ')
    if len(mappings) > 1:
        seq = ''
        score = old_score
        print(score, end=', ')
        length = cipher_text.rfind(cipher_letter) + 1
        for i in range(0, length):
            c = deciphered[i]
            if cipher_text[i] == cipher_letter:
                # Calculate score for the newly ciphered letters using nlm
                # Note that nlm scores are positive (-log(Prob)).
                if i == 0:
                    score += nlm.score_first(c, model, cuda)
                else:
                    score += nlm.score_next(c, seq, model, cuda)
                print(seq, score, end=', ')
            elif c == '_':
                # Predict unknown letters
                if i == 0:
                    # First letter of Zodiac408 is i
                    c = 'i'
                else:
                    llh_predictions = nlm.next_chars(seq[:i], cuda, model, k=5, cutoff='symbol')
                    if llh_predictions:
                        # Last element has the lowest nlm score and highest prob
                        c = llh_predictions[-1][0]
                    else:
                        # nlm gives no prediction. Use ngram instead.
                        # Note that ngram scores are negative (log(Prob)).
                        highest_score = 0
                        optimal_e = None
                        for e in Ve:
                            cur_score = lm.score_seq(seq[1-lm_order:] + e)
                            if -cur_score > highest_score:
                                highest_score = -cur_score
                                optimal_e = e
                        c = optimal_e
            seq = seq + c
    else:
        # With just one letter deciphered, we score using ngram to speed it up.
        bitstring = ['.' if c == '_' else 'o' for c in deciphered]
        deciphered = ''.join(deciphered)
        bitstring = ''.join(bitstring)
        # Note that ngram scores are negative (log(Prob)).
        score = -lm.score_bitstring(deciphered, bitstring)

    print(score, end=', ')
    # frequency matching heuristic
    score += math.fabs(math.log(cipher_freq[cipher_letter] / plaintxt_freq[mappings[cipher_letter]]))
    print(score)

    return score


def prune(beams, beamsize):
    sorted_beams = sorted(beams, key=lambda b: b[1])

    return sorted_beams[:beamsize]


def beam_search(cipher_text, ext_order, Ve, ext_limits, init_beamsize, lm_order, lm, model, cuda,
                cipher_freq, plaintxt_freq):
    Hs = []
    Ht = []
    cardinality = 0
    Hs.append(({}, 0))

    while cardinality < len(ext_order):
        beamsize = int(init_beamsize*(0.94**cardinality))
        print("Searching for {}/{} letter".format(cardinality, len(ext_order)))
        print("\tCurrent size of searching tree: {:,}".format(len(Hs)))
        print("\tGoing to be expended to: {:,}".format(len(Hs) * len(Ve)))
        cipher_letter = ext_order[cardinality]
        for mappings, sc in Hs:
            for plain_letter in Ve:
                ext_mappings = deepcopy(mappings)
                ext_mappings[cipher_letter] = plain_letter
                if check_limits(ext_mappings, ext_limits, plain_letter):  # only check new added one
                    Ht.append((ext_mappings, score(Ve, ext_mappings, cipher_text, cipher_letter, sc, 
                                                   lm_order, lm, model, cuda, cipher_freq, plaintxt_freq)))
        Hs = prune(Ht, beamsize)
        cardinality += 1
        Ht = []
    Hs.sort(key=lambda b: b[1], reverse=True)
    return Hs[0]


def contiguous_score(cipher, order, lm_order, weights):
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
        score += weights[k] * v
    return score


def prune_orders(orders, beamsize):
    sorted_order = sorted(orders, reverse=True)

    return sorted_order[: beamsize]


def search_ext_order(cipher, beamsize, lm_order, contiguous_score_weights):
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
                    new_scores.insert(0, contiguous_score(cipher, new_order, lm_order, contiguous_score_weights))
                    orders_tmp.append((new_scores, new_order))
        orders = prune_orders(orders_tmp, beamsize)
        orders_tmp = []
    orders.sort(reverse=True)
    return orders[0][1]

def get_statistics(content, cipher=True):
    stats = {}
    content = list(content)
    split_content = [x for x in content if x != '\n' and x!=' ']
    length = len(split_content)
    symbols = set(split_content)
    uniq_sym = len(list(symbols))
    freq = collections.Counter(split_content)
    rel_freq = {}
    for sym, frequency in freq.items():
        rel_freq[sym] = (frequency/length)*100

    if cipher:
        stats = {'content':split_content, 'length':length, 'vocab':list(symbols), 'vocab_length':uniq_sym, 'frequencies':freq, 'relative_freq':rel_freq}
    else:
        stats = {'length':length, 'vocab':list(symbols), 'vocab_length':uniq_sym, 'frequencies':freq, 'relative_freq':rel_freq}
    return stats

if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-f', '--file', default='data/cipher.txt', help='cipher file')
    arg_parser.add_argument('-b', '--beamsize', type=int, default=1000)
    arg_parser.add_argument('--cuda', action='store_true', default=False)
    arg_parser.add_argument('-el', '--ext_limits', type=int, default=7)
    args = arg_parser.parse_args()

    lm_order = 6
    contiguous_score_weights = [0,0,1,1,1,2,3]

    Ve = string.ascii_lowercase

    print('Loading language model')
    lm = LM("data/6-gram-wiki-char.lm.bz2", n=lm_order, verbose=False)
    model = nlm.load_model("data/mlstm_ns.pt", cuda=args.cuda)
    print('Language model loaded')

    cipher = read_file(args.file)
    cipher = [x for x in cipher if not x.isspace()]
    cipher = ''.join(cipher)

    ext_order = search_ext_order(cipher, 100, lm_order, contiguous_score_weights)
    print(ext_order)

    cipher_freq = get_statistics(cipher, cipher=True)['relative_freq']
    plaintxt = read_file("data/default.wiki.txt.bz2")
    plaintxt_freq = get_statistics(plaintxt, cipher=False)['relative_freq']

    print('Start deciphering...')
    search_start = datetime.now()
    mappings, sc = beam_search(cipher, ext_order, Ve, args.ext_limits, args.beamsize,
                               lm_order, lm, model, args.cuda,
                               cipher_freq, plaintxt_freq)
    search_end = datetime.now()
    print('Deciphering completed after {}'.format(search_end - search_start))

    print(mappings)
    deciphered = [mappings[c] if c in mappings else '_' for c in cipher]
    deciphered = ''.join(deciphered)
    print(deciphered, sc)
