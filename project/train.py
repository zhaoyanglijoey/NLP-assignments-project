import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from dataloader import TweetsDataset
import pandas as pd
import torch.optim as optim

class TwitterSentiment():
    def __init__(self, train_file, test_file):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.train_data = pd.read_csv(train_file)
        self.test_data = pd.read_csv(test_file)
        self.train_set = TweetsDataset(train_file, tokenizer)
        self.test_data = TweetsDataset(test_file, tokenizer)
        self.train_loader = DataLoader(self.train_set, batch_size=1, shuffle=True)
        self.test_loader = DataLoader(self.test_data, batch_size=1)


        self.optimizer = optim.Adam()