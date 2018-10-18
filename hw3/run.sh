# bash

#source venv/bin/activate
python count-sentences.py -i data/train.txt.gz

echo "training..."
python default.py -m data/default.model

echo "validating..."
python perc.py -m data/default.model > output

python score_chunks.py -t output


