# HW1


## Parsing tree to grammar

### Issues

- Non terminals are supposed to be all upper case letters, 
while terminals are supposed to be not all upper case letters.
but some terminals have one single upper case letter.
For example 'B' is a terminal. This makes it hard to tell 
vocabularies(terminals) from non terminals. 

- Parenthesis exists in the given text. For example in line 410 of `devset.txt`:
`( Fetchez la vache . ) wha ?`. This results in unmatched parenthesis in `devset.tree`
and empty non terminal and terminal in the parsed grammar. We should consider 
substitute the parenthesis with other symbol when generating grammar tree.

### Tasks

- [x] Parse grammar based on frequency of appearance in the devset.tree
- [ ] Convert grammars to CNF
- [ ] Make it work!
- [ ] Improve?  