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
arg_parser.add_argument('--no-decay', action='store_true', default=False)

args = arg_parser.parse_args()

lm_order = 6
contiguous_score_weights = [0,0,1,1,1,2,3]

# lm_order = 20
# contiguous_score_weights = [0,0,1,1,1,2,3,4,5,6,7,  8,9,10,11,12, 13,14,15,16,17 ]

ext_limits = {letter: 4 if letter is not 'e' else 7 for letter in string.ascii_lowercase}

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
        return plaintext_letters.count(letter_to_check) <= ext_limits[letter_to_check]

def score_single_seq(t):
    i, seq = t

    return len(seq) * ( lm.score_partial_seq(seq) if i != 0 else lm.score_seq(seq) )

pool = Pool(args.num_workers)

def score(mappings, cipher_text, lm, nlm):
    deciphered = [mappings[cipher_letter] if cipher_letter in mappings else ' ' for cipher_letter in cipher_text]
    deciphered = ''.join(deciphered)
    # bit_string = [ 'o' if c in mappings else '.' for c in cipher_text]
    # bit_string = ''.join(bit_string)
    seqs = deciphered.split()
    seqs = list(filter(lambda seq: len(seq) > 2, seqs))

    res = sum(pool.map(score_single_seq, zip(range(len(seqs)),seqs)))

    # return lm.score_bitstring(deciphered, bit_string)
    return res

def prune(beams, beamsize):
    sorted_beams = sorted(beams, key=lambda b: b[1], reverse=True)

    return sorted_beams[:beamsize]

def get_true_mappings(cipher):
    with open('data/ref.txt') as f:
        ref = f.read()
    true_mappings = {}
    num_symbols = len(set(cipher))
    for i in range(len(cipher)):
        if cipher[i] not in true_mappings:
            true_mappings[cipher[i]] = ref[i]
            if len(true_mappings) == num_symbols:
                return true_mappings

def decipher(cipher, mappings):
    deciphered = [mappings[cipher_letter] if cipher_letter in mappings else '.' for cipher_letter in cipher]
    deciphered = ''.join(deciphered)
    return deciphered

def beam_search(cipher_text, lm, nlm, ext_order, ext_limits, beamsizes):
    Hs = []
    Ht = []
    cardinality = 0
    Hs.append(({}, 0))
    Ve = string.ascii_lowercase

    true_mappings = get_true_mappings(cipher)

    while cardinality < len(ext_order):
        # if args.no_decay:
        #     beamsize = init_beamsize
        # else:
        #     beamsize = max(100, int(init_beamsize*(0.95**cardinality)))
        beamsize = beamsizes[cardinality]

        print("Searching for {}/{} letter".format(cardinality, len(ext_order)))
        print("\tCurrent size of searching tree: {:,}".format(len(Hs)))
        # print("\tGoing to be expended to: {:,}".format(len(Hs) * len(Ve)))
        cipher_letter = ext_order[cardinality]
        for mappings, sc in Hs:
            for plain_letter in Ve:
                ext_mappings = deepcopy(mappings)
                ext_mappings[cipher_letter] = plain_letter
                if check_limits(ext_mappings, ext_limits, plain_letter):  # only check new added one
                    Ht.append((ext_mappings, score(ext_mappings, cipher_text, lm, nlm)))
        Hs = prune(Ht, beamsize)
        max_acc, acc_deciphered = check_gold(Hs, cipher_text)
        print("Check gold: the best accuracy is: {}\nDeciphered text: \n{}".format(max_acc, acc_deciphered))
        # print("\tMost likely plaintext: \n{}".format(decipher(cipher_text, Hs[0][0])))
        cardinality += 1
        Ht = []
        best_mappings = Hs[0][0]
        best_sc = Hs[0][1]
        best_deciphered = decipher(cipher, best_mappings)

        worst_mappings = Hs[-1][0]
        worst_sc = Hs[-1][1]
        worst_deciphered = decipher(cipher, worst_mappings)

        true_deciphered = [true_mappings[cipher_letter] if cipher_letter in best_mappings else '.' for cipher_letter in cipher]
        true_deciphered = ''.join(true_deciphered)
        seqs = true_deciphered.replace('.', ' ') .split()
        seqs = list(filter(lambda seq: len(seq) > 2, seqs))
        true_score = sum(pool.map(score_single_seq, zip(range(len(seqs)), seqs)))

        print('Best deciphered text: \n{} score: {} \nTrue text: \n{} score: {}\nWorst deciphered text: \n{} score: {}\n'
              .format(best_deciphered, best_sc, true_deciphered, true_score, worst_deciphered, worst_sc))
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
            ngrams[min(lm_order, count)] += 1
        else:
            count = 0

    score = 0
    for k, v in ngrams.items():
        score += contiguous_score_weights[k] * v
    return score

def prune_orders(orders, beamsize):
    sorted_order = sorted(orders, reverse=True)

    return sorted_order[: beamsize]

# def search_ext_order(cipher, beamsize):
#     symbols = set(cipher)
#     order = []
#     for c in cipher:
#         if c not in order:
#             order.append(c)
#             if len(order) == len(symbols):
#                 return order

def search_ext_order(cipher, beamsize):
    symbols = set(cipher)
    # Start with the most common character
    freq = Counter(cipher)
    start = freq.most_common(1)[0][0]
    # start = cipher[0]
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
    return orders[0][1]

#
# def decipher(mappings, cipher_text):
#     deciphered = [mappings[c] if c in mappings else '_' for c in cipher_text]
#     deciphered = ''.join(deciphered)
#     return deciphered


def check_gold(Hs, cipher_text):
    """
    Each iteration, check whether current best solutions. (In order to check in which step the the solution is pruned)
    :param Hs:
    :param cipher_text:
    :return: max acc
    """
    max_acc = 0
    deciphered_text = None
    for mappings, sc in Hs:
        deciphered = decipher(cipher_text, mappings)
        if max_acc < evaluator.evaluate(deciphered):
            max_acc = evaluator.evaluate(deciphered)
            deciphered_text = deciphered
    return max_acc, deciphered_text

def dynamic_beamsize(cipher, beamsize):
    num_symbols = len(set(cipher))
    beamsizes = [beamsize] * (num_symbols)
    # for i in range(4):
    #     beamsizes[i] = 1000000
    # beamsizes[10] = 300000
    # beamsizes[20] = 300000
    # for i in range(num_symbols // 2, num_symbols):
    #     beamsizes[i] = int(beamsize * (0.85 ** (i - num_symbols//2)))
    return beamsizes

if __name__ == '__main__':

    cipher = read_file(args.file)
    cipher = [x for x in cipher if not x.isspace()]
    cipher = ''.join(cipher)
    # freq = Counter(cipher)
    # ext_order = [ kv[0] for kv in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)]

    ext_order = search_ext_order(cipher, 50)
    beamsizes = dynamic_beamsize(cipher, args.beamsize)
    print(ext_order)

    print('Start deciphering...')
    search_start = datetime.now()
    mappings, sc = beam_search(cipher, lm, nlm, ext_order, ext_limits, beamsizes)

    search_end = datetime.now()
    print('Deciphering completed after {}'.format(search_end - search_start))
    print(mappings)
    deciphered = decipher(cipher, mappings)
    print(deciphered, sc)
    print(evaluator.evaluate(deciphered, log=True))