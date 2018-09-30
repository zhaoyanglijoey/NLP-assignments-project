import nltk
import argparse


"""
Given a vocab file, give a allowed words file.
Append unseen words to the end of vocab file with default tag
"""


def main(vocab_file, allowed_words_file, default_weight):
    unseen_words = set()
    with open(allowed_words_file, "r") as f:
        for line in f:
            word = line.strip()
            unseen_words.add(word)

    with open(vocab_file, "r") as f:
        for line in f:
            word = line.split()[2]
            if word in unseen_words:
                unseen_words.remove(word)

    vocab_grammar = handle_unseen_words(unseen_words, default_weight)

    with open(vocab_file, "a") as f:
        # append to the end of vocab file
        f.write("\n")
        f.write("\n".join(vocab_grammar))

    print("Append {} words to {}".format(len(vocab_grammar), vocab_file))


def handle_unseen_words(unseen_words, default_weight):
    vocab_grammar = []
    # format: {:<8} {:<8} {}
    for unseen_word in unseen_words:
        pos_tag = get_tag(unseen_word)
        vocab_grammar.append("{:<8} {:<8} {}".format(default_weight, pos_tag, unseen_word))

    return vocab_grammar


def get_tag(word):
    tag = nltk.tag.pos_tag([word])[0][1]
    return tag


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--vocab", dest="vocab_file", required=True, help="vocab gr file")
    ap.add_argument("-a", "--allowed-words", dest="allowed_words_file", required=True, help="allowed words file")
    ap.add_argument("-w", "--weight", dest="default_weight", default=1, help="default weight for unseen words")
    args = ap.parse_args()
    main(args.vocab_file, args.allowed_words_file, args.default_weight)
