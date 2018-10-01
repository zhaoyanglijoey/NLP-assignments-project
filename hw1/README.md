# HW1


## Parsing tree to grammar

### How to use

Edit input tree location & output grammar locations in side `generate_grammar_from_tree.sh`
Then run the script
```
./generate_grammar_from_tree.sh
```


### Issues

- Non terminals are supposed to be all upper case letters, 
while terminals are supposed to be not all upper case letters.
but some terminals have one single upper case letter.
For example 'B' is a terminal. This makes it hard to tell 
vocabularies(terminals) from non terminals. 

- <del>Parenthesis exists in the given text. For example in line 410 of `devset.txt`:
`( Fetchez la vache . ) wha ?`. This results in unmatched parenthesis in `devset.tree`
and empty non terminal and terminal in the parsed grammar. We should consider 
substitute the parenthesis with other symbol when generating grammar trees.</del> 
Seems like it is solved by our text parser that generates grammar trees.

- How to test S2.gr? Based on some experiments, when the train set is large enough, and all words are added to Vocab.gr,
 all sentences can be handled by S1.gr


### Tasks

- [x] Parse grammar based on frequency of appearance in the devset.tree
- [x] Convert grammars to CNF
- [x] Make it work!
- [x] Add unseen words to vocab, use nltk to assign pos tag
- [ ] Add & test backoff support
- [ ] Improve?  