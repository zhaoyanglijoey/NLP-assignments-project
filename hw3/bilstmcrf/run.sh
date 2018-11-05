# bash

python ../count-sentences.py -i ../data/train.txt.gz

echo "training..."
python bilstmcrf_trainer.py -m bilstmcrf.model

echo "testing..."
python bilstmcrf_tester.py -m bilstmcrf.model > output

echo "scoring..."
cd ..
python score_chunks.py -t bilstmcrf/output
