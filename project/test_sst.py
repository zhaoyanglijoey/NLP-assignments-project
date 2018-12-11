import os, argparse, glob2, pickle
from pytorch_pretrained_bert import BertForSequenceClassification, BertTokenizer
import torch
from clean_data import clean_tweet
import util
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.functional import softmax
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json


if __name__ == '__main__':
    path = os.path.join('data', 'sst_test.tsv')
    df = pd.read_csv(path, encoding='latin1', usecols=[0, 1], sep='\t', quoting=3)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--load')

    args = argparser.parse_args()

    print('creating model')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(args.load))
    model.to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    output = open('SST-2.tsv', 'w')
    output.write('id\tlabel\n')
    for row in df.itertuples():
        seq = str(row[2])
        idx = str(row[1])
        id, mask = util.convert_to_bert_ids(seq, tokenizer, 100)
        id = id.unsqueeze(0)
        mask = mask.unsqueeze(0)
        logits = model(id, attention_mask=mask)
        pred = torch.argmax(logits[0])
        print('{}\t{}'.format(idx, pred))
        output.write('{}\t{}\n'.format(idx, pred))
    output.close()


