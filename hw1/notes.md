1. ### Baseline

> The baseline method for `S1.gr` is to use `example-sentences.txt` and `devset.txt` to create a weighted context-free grammar from the parse trees (some trees are provided in `devset.trees`). 
>
> You might want to augment this data with other sentences from the same distribution. Also, see the section on sharing samples below.
>
> You can improve the grammar by iteratively extracting a grammar and improving the weighted grammar. If you can find more sentences to parse, use an automatic parser that can produce parse trees from the [Penn Treebank](https://catalog.ldc.upenn.edu/ldc99t42). Note that you do not need access to the treebank, you just need a parser that was trained on it.
>
> For `S2.gr` a good baseline is to exploit how often a word follows another word in `example-sentences.txt` and `devset.txt` or sentences sampled from other group grammars to produce the probability for generating a sentence.
>
> For `Vocab.gr` the baseline should at least cover all the words in `allowed_words.txt` and you should either provide or infer the part of speech or pre-terminal non-terminal category.

Tasks:

1. Extract grammer from `devset.txt` (Voca, S1)
2. Find more sentences & generate trees
3. Find a state of art parser trained on the Penn Treebank