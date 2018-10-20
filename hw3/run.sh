# bash

#source venv/bin/activate
python count-sentences.py -i data/train.txt.gz

echo "training..."
python default.py -m data/default.model -e 5

echo "validating..."
python tester.py -m data/default.model > output

python score_chunks.py -t output


