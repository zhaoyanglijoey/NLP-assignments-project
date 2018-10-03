from nltk import Tree
from collections import defaultdict
import argparse

ROOT_NODE_NAME = "ROOT"

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
                lastch = None
                i = 0
                while i < len(data):
                    ch = data[i]
                    if lastch == '(' and ch == '(':
                        treestr += '-LRB- -LRB-'
                        i += 3
                        lastch = None
                        continue
                    if lastch == '(' and ch == ')':
                        treestr += '-RRB- -RRB-'
                        i += 3
                        lastch = None
                        continue

                    treestr += ch
                    if ch == '(':
                        count += 1
                    elif ch == ')':
                        count -= 1
                        if count == 0:
                            self.parse_treestr(treestr)
                            treestr = ''
                    lastch = ch
                    i += 1

    def parse_treestr(self, treestr):
        treestr = treestr.strip()
        tree = Tree.fromstring(treestr)
        if tree.label() != ROOT_NODE_NAME:
            new_root = Tree.fromstring(f"({ROOT_NODE_NAME})")
            new_root.insert(0, tree)
            tree = new_root
        tree.chomsky_normal_form()
        self.starts[tree.label()] += 1
        # print(tree)
        # tree.pretty_print()
        self.traverse_tree(tree)

    def traverse_tree(self, tree):
        tree.set_label(self.sanitize_nont(tree.label()))
        rhs = ''
        for subtree in tree:
            if type(subtree) == Tree:
                self.traverse_tree(subtree)
                rhs += subtree.label() + ' '
            else:
                rhs += self.sanitize_t(subtree)
        # if tree.label()=='':
        #     tree.pretty_print()
        #     print('setence:', ' '.join(tree.leaves()))
        self.grammar[tree.label()][rhs.strip()] += 1
        # tree.pretty_print()
        # print(tree.label(), '->', rhs)

    def sanitize_nont(self, nont):
        nont = nont.strip()
        if nont == '.':
            nont = 'PERIOD'
        if nont == ':':
            nont = 'COLON'
        if nont == ',':
            nont = 'COMMA'
        if nont == "''":
            nont = 'TWOSINGLEQUOTES'
        if nont == "``":
            nont = 'TWOGRAVES'
        return nont

    def sanitize_t(self, t):
        t = t.strip()
        if t == '-LRB-':
            t = '('
        if t == '-RRB-':
            t = ')'
        return t

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
        s1_out += '{:<8} {:<8} S1\n'.format('99', 'TOP')
        s1_out += '{:<8} {:<8} S2\n'.format('1', 'TOP')
        for start, freq in self.starts.items():
            s1_out += '{:<8} {:<8} {}\n'.format(freq, 'S1', start)
        vocab_out = ''
        for lhs, grammar_dict in self.grammar.items():
            if lhs == '':
                continue

            for rhs, freq in grammar_dict.items():
                if rhs == '':
                    continue
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
    arg_parser.add_argument('-i', dest='gtrees', nargs='+', help='input grammar trees')
    arg_parser.add_argument('-os1', dest='s1', required=True, help='output s1')
    arg_parser.add_argument('-ov', dest='vocab', required=True, help='output vocab')
    args = arg_parser.parse_args()

    treeParser = TreesParser()
    treeParser.parse(args.gtrees)
    treeParser.to_grammar(args.s1, args.vocab)


if __name__ == '__main__':
    main()



