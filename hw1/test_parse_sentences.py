from pcfg_parse_gen import Pcfg, PcfgGenerator, CkyParse
import nltk


def print_tree(tree_string):
    tree_string = tree_string.strip()
    tree = nltk.Tree.fromstring(tree_string)
    tree.pretty_print()


def draw_tree(tree_string):
    tree_string = tree_string.strip()
    tree = nltk.Tree.fromstring(tree_string)
    tree.draw()


def main():
    sent = "Sir Lancelot it ."
    parse_gram = Pcfg(["dev_s1.gr", "S2.gr", "dev_vocab.gr"])
    # parse_gram = Pcfg(["S1.gr", "S2.gr", "dev_vocab.gr"])
    parser = CkyParse(parse_gram, beamsize=0.0001)
    ce, trees = parser.parse_sentences([sent])
    print("-cross entropy: {}".format(ce))
    for tree_string in trees:
        print_tree(tree_string)


if __name__ == '__main__':
    main()
