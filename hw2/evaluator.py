import string


def read_gold(gold_file):
    with open(gold_file) as f:
        gold = f.read()
    f.close()
    gold = list(gold.strip())
    return gold


def symbol_error_rate(dec, _gold):
    gold = read_gold(_gold)
    n = len(gold)
    correct = 0
    if len(gold) == len(dec):
        for (d, g) in zip(dec, gold):
            if d == g:
                correct += 1
            elif d not in string.ascii_lowercase:
                n -= 1

    wrong = n - correct
    error = wrong / n

    return error


def evaluate(plaintext, log=False):
    # gold decipherment
    gold_file = "data/ref.txt"
    ser = symbol_error_rate(plaintext, gold_file)
    if log:
        print('Error: ', ser * 100, 'Accuracy: ', (1 - ser) * 100)
    return int((1 - ser) * 100)
