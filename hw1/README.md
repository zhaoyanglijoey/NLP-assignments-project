# HW1


## Parsing tree to grammar

### How to use

```
python3 parse_tree.py -i <grammar_trees> [<grammar_trees> ...] 
```

Eg. `python3 parse_tree.py -i devset.trees`  

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

### Tasks

- [x] Parse grammar based on frequency of appearance in the devset.tree
- [x] Convert grammars to CNF
- [x] Make it work!
- [ ] Add backoff support
- [ ] Improve?  