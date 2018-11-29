import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertAdam, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader, TensorDataset
from dataloader import TweetsDataset
import pandas as pd
import torch.optim as optim
from util import convert_data_to_features
from tqdm import tqdm
import numpy as np


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


class TwitterSentiment():
    def __init__(self, train_file, test_file, batch_size=32, num_epoch=10, log_interval=100, prototype=False):
        self.log_interval = log_interval
        self.batch_size = batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.train_data = pd.read_csv(train_file)
        self.test_data = pd.read_csv(test_file)
        if prototype:
            self.train_data = self.train_data[:10000]
            self.test_data = self.test_data[:1000]
        # self.train_set = TweetsDataset(train_file, tokenizer)
        # self.test_data = TweetsDataset(test_file, tokenizer)
        train_features = convert_data_to_features(self.train_data, [0, 1], 200, tokenizer)
        train_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        train_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        train_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        self.train_set = TensorDataset(train_input_ids, train_input_mask, train_label_ids)
        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)

        test_features = convert_data_to_features(self.test_data, [0, 1], 200, tokenizer)
        test_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        test_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        test_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
        self.test_set = TensorDataset(test_input_ids, test_input_mask, test_label_ids)
        self.test_loader = DataLoader(self.test_set, batch_size=batch_size)

        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        self.model.to(self.device)

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
        self.num_epoch = num_epoch
        t_total = int(len(self.train_data) / batch_size * num_epoch)
        self.optimizer = BertAdam(optimizer_grouped_parameters, 5e-5, warmup=0.1, t_total=t_total)

    def train_epoch(self, epoch):
        self.model.train()
        running_ls = 0
        for i, batch  in enumerate(tqdm(self.train_loader)):
            (input_ids, input_mask, label_ids) = tuple(t.to(self.device) for t in batch)

            self.model.zero_grad()
            loss, _ = self.model(input_ids, attention_mask=input_mask, labels=label_ids)
            loss.backward()
            running_ls += loss.item()
            self.optimizer.step()

            if (i+1) % self.log_interval == 0:
                print('[{:>3}, {:>7}] loss:{:.4}'.format(epoch, (i+1)*self.batch_size, running_ls / self.log_interval))
                running_ls = 0

    def test(self):
        self.model.eval()
        eval_loss = 0
        eval_accuracy = 0
        count = 0
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.test_loader)):
                count += 1
                (input_ids, input_mask, label_ids) = tuple(t.to(self.device) for t in batch)
                loss, logits = self.model(input_ids, attention_mask=input_mask, labels=label_ids)
                eval_loss += loss.item()
                logits = logits.cpu.numpy()
                label_ids = label_ids.cpu.numpy()
                eval_accuracy += accuracy(logits, label_ids)

        eval_loss /= count
        eval_accuracy /= len(self.test_loader)
        print('evaluation loss: {:.4}, accuracy: {:.4}%'.format(eval_loss, eval_accuracy * 100))


if __name__ == '__main__':
    train_file = 'data/train.csv'
    test_file = 'data/test.csv'
    num_epoch = 10
    twitter_sentiment = TwitterSentiment(train_file, test_file, num_epoch=num_epoch, prototype=True)
    for e in range(num_epoch):
        twitter_sentiment.train_epoch(e+1)
        twitter_sentiment.test()
