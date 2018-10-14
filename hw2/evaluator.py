def read_gold(gold_file):
    with open(gold_file) as f:
        gold = f.read()
    f.close()
    gold = list(gold.strip())
    return gold


def symbol_error_rate(dec, _gold):
    gold = read_gold(_gold)
    correct = 0
    if len(gold) == len(dec):
        for (d, g) in zip(dec, gold):
            if d == g:
                correct += 1
    wrong = len(gold) - correct
    error = wrong / len(gold)

    return error


def evaluate(plaintext):
    # gold decipherment
    gold_file = "data/ref.txt"
    ser = symbol_error_rate(plaintext, gold_file)
    print('Error: ', ser * 100, 'Accuracy: ', (1 - ser) * 100)