### Homework 3: Phrasal Chunking

This is for the assignment 3 of CMPT413 in Fall 2018.

#### How to run

Run `./bilstmcrf.sh`:

```
# Count sentences.
python count-sentences.py -i data/train.txt.gz

# Model will be trained and saved to ./bilstmcrf.model by chunk.py
python chunk.py -m bilstmcrf.model

# Output on input file will be saved to ./output by bilstmcrf_tester.py
# Use --inputfile (-i) and --featfile (-f) to specify input txt file and feat file.
# By default, inputfile="data/dev.txt" and featfile="data/dev.feats".
python bilstmcrf_tester.py -m bilstmcrf.model > output

# ./output will be scored by score_chunks.py
python score_chunks.py -t output
```