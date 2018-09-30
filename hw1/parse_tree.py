import nltk
from nltk import Tree
from collections import defaultdict

class TreesParser():
    def __init__(self):
        self.grammar = defaultdict(lambda: defaultdict(int))
        self.starts = defaultdict(int)

    def parse(self, filename):
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
        s1_out = ''
        s1_out += '{:<8} {:<8} S1\n'.format('100', 'TOP')
        for start, freq in self.starts.items():
            s1_out += '{:<8} {:<8} {}\n'.format(freq, 'S1', start)
        vocab_out = ''
        for lhs, dict in self.grammar.items():
            if lhs =='':
                continue

            for rhs, freq in dict.items():
                if rhs.isupper():
                    s1_out += '{:<8} {:<8} {}\n'.format(freq, lhs, rhs)
                else:
                    vocab_out += '{:<8} {:<8} {}\n'.format(freq, lhs, rhs)

        with open(s1_filename, 'w') as f:
            f.write(s1_out)
        with open(vocab_filename, 'w') as f:
            f.write(vocab_out)

filename = 'devset.trees'
treeParser = TreesParser()
treeParser.parse(filename)
treeParser.to_grammar('dev_s1.gr', 'dev_vocab.gr')


