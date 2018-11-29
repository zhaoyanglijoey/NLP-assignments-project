import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
from torch.utils.data import Dataset
import pandas as pd


class TweetsDataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

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
        label = torch.tensor([entry['tag']], dtype=torch.long)
        tweet = entry['cleaned_tweet']
        tokenized_tweet = self.tokenizer.tokenize(tweet)
        if len(tokenized_tweet) > self.max_seq_length - 2:
            tokenized_tweet = tokenized_tweet[0:(self.max_seq_length - 2)]

        tokenized_tweet.insert(0, '[CLS]')
        tokenized_tweet.append('[SEP]')
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_tweet)

        token_ids = [0] * self.max_seq_length
        token_ids[:len(indexed_tokens)] = indexed_tokens
        input_mask = [0] * self.max_seq_length
        input_mask[:len(indexed_tokens)] = 1

        assert len(token_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length

        token_ids = torch.tensor(token_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)

        return token_ids, input_mask, label