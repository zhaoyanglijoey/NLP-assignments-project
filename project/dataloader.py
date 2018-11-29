import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
from torch.utils.data import Dataset
import pandas as pd


class TweetsDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        '''
        :param index:
        :return:
            img: [ , 224, 224, 3] tensor
            word_ind: [ , T] word indices tensor
        '''

        entry = self.data.iloc[index]
        tag = torch.tensor([entry['tag']], dtype=torch.long)
        tweet = entry['cleaned_tweet']
        tokenized_tweet = self.tokenizer.tokenize(tweet)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_tweet)
        tokens_tensor = torch.tensor([indexed_tokens])

        return tokens_tensor, tag