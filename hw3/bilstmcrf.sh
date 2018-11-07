# bash

python count-sentences.py -i data/train.txt.gz

echo "training..."
python chunk.py -m bilstmcrf.model

echo "testing..."
python bilstmcrf_tester.py -m bilstmcrf.model > output

echo "scoring..."
python score_chunks.py -t output
