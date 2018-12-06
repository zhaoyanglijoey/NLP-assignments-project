import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
from torch.utils.data import Dataset
import pandas as pd
import sys
import util

class TweetsDataset(Dataset):
    def __init__(self, data, tokenizer, length_limit):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = length_limit

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        '''
        :param index:
        :return:
            token_ids: index of tokens
            input_mask:
            label: GT label
        '''

        entry = self.data.iloc[index]
        label = torch.tensor(entry['tag'], dtype=torch.long)
        tweet = str(entry['cleaned_tweet'])
        ids, mask = util.convert_to_bert_ids(tweet, self.tokenizer, self.max_seq_length)

        return ids, mask, label
