from nltk import Tree
from collections import defaultdict
import argparse

class TreesParser():
    def __init__(self):
        self.grammar = defaultdict(lambda: defaultdict(int))
        self.starts = defaultdict(int)

    def parse(self, files):
        for filename in files:
            with open(filename) as f:
                data = f.read()
                count = 0
                treestr = ''
                for ch in data:
                    treestr += ch
                    if ch == '(':
                        count += 1
                    elif ch == ')':
                        count -= 1
                        if count == 0:
                            self.parse_treestr(treestr)
                            treestr = ''

    def parse_treestr(self, treestr):
        treestr = treestr.strip()
        tree = Tree.fromstring(treestr)
        tree.chomsky_normal_form()
        self.starts[tree.label()] += 1
        self.traverse_tree(tree)

    def traverse_tree(self, tree):
        rhs = ''
        for subtree in tree:
            if type(subtree) == Tree:
                rhs += subtree.label() + ' '
                self.traverse_tree(subtree)
            else:
                rhs += subtree
        # if tree.label()=='':
        #     tree.pretty_print()
        #     print('setence:', ' '.join(tree.leaves()))
        self.grammar[tree.label()][rhs.strip()] += 1
        # tree.pretty_print()
        # print(tree.label(), '->', rhs)

    def to_grammar(self, s1_filename, vocab_filename):
        def _add_other_allowed_words(grammar_dict: dict):
            appended_grammar = ""
            default_freq = 1
            default_pos = 'Misc'
            with open('allowed_words.txt', 'r') as f:
                for line in f:
                    word = line.strip()
                    if word not in grammar_dict.values() or 1:
                        appended_grammar += '{:<8} {:<8} {}\n'.format(default_freq, default_pos, word)
            return appended_grammar

        s1_out = ''
        s1_out += '{:<8} {:<8} S1\n'.format('100', 'TOP')
        # s1_out += '{:<8} {:<8} S2\n'.format('1', 'TOP')
        for start, freq in self.starts.items():
            s1_out += '{:<8} {:<8} {}\n'.format(freq, 'S1', start)
        vocab_out = ''
        for lhs, grammar_dict in self.grammar.items():
            if lhs == '':
                continue

            for rhs, freq in grammar_dict.items():
                if rhs.isupper():
                    s1_out += '{:<8} {:<8} {}\n'.format(freq, lhs, rhs)
                else:
                    vocab_out += '{:<8} {:<8} {}\n'.format(freq, lhs, rhs)

        # appended_grammar = _add_other_allowed_words(grammar_dict)
        # vocab_out += appended_grammar

        with open(s1_filename, 'w') as f:
            f.write(s1_out)
        with open(vocab_filename, 'w') as f:
            f.write(vocab_out)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-i',dest='gtrees', nargs='+', help='input grammar trees')
    args = arg_parser.parse_args()

    treeParser = TreesParser()
    treeParser.parse(args.gtrees)
    treeParser.to_grammar('dev_s1.gr', 'dev_vocab.gr')

if __name__ == '__main__':
    main()



