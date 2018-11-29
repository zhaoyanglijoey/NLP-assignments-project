import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertAdam, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader, TensorDataset
from dataloader import TweetsDataset
import pandas as pd
import torch.optim as optim
from util import convert_data_to_features, check_path
from tqdm import tqdm
import numpy as np
import time

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def f1score(out, labels):
    outputs = np.argmax(out, axis=1)
    tp = outputs == labels


class TwitterSentiment():
    def __init__(self, train_file, test_file, batch_size=8, num_epoch=10, log_interval=100, prototype=False):
        self.log_interval = log_interval
        self.batch_size = batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.train_data = pd.read_csv(train_file, dtype={'tag':int, 'cleaned_tweet':str})
        self.test_data = pd.read_csv(test_file, dtype={'tag':int, 'cleaned_tweet':str})
        if prototype:
            self.train_data = self.train_data[:5000]
            self.test_data = self.test_data[:500]
        self.train_set = TweetsDataset(self.train_data, tokenizer, 200)
        self.test_set = TweetsDataset(self.test_data, tokenizer, 200)

        # train_features = convert_data_to_features(self.train_data, [0, 1], 200, tokenizer)
        # train_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        # train_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        # train_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        # self.train_set = TensorDataset(train_input_ids, train_input_mask, train_label_ids)
        #
        # test_features = convert_data_to_features(self.test_data, [0, 1], 200, tokenizer)
        # test_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        # test_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        # test_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
        # self.test_set = TensorDataset(test_input_ids, test_input_mask, test_label_ids)

        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)
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
        start = time.time()
        num_batches = len(self.train_loader)
        for i, batch  in enumerate(self.train_loader):
            (input_ids, input_mask, label_ids) = tuple(t.to(self.device) for t in batch)

            self.model.zero_grad()
            loss, _ = self.model(input_ids, attention_mask=input_mask, labels=label_ids)
            loss.backward()
            running_ls += loss.item()
            self.optimizer.step()

            if (i+1) % self.log_interval == 0:
                elapsed = time.time() - start
                iters_per_sec = (i+1) / elapsed
                remaining = (num_batches - i - 1) / iters_per_sec
                remaining_h = int(remaining // 3600)
                remaining_m = int(remaining // 60 % 60)
                remaining_s = int(remaining % 60)
                print('[{:>3}, {:>7}/{}] loss:{:.4}  {:.3}iters/s {:02}:{:02}:{:02} left'.format(epoch, (i+1), num_batches,
                        running_ls / self.log_interval, iters_per_sec, remaining_h, remaining_m, remaining_s))
                running_ls = 0

    def test(self):
        self.model.eval()
        eval_loss = 0
        eval_accuracy = 0
        batches_count = 0
        data_count = 0
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.test_loader)):
                batches_count += 1
                (input_ids, input_mask, label_ids) = tuple(t.to(self.device) for t in batch)
                data_count += input_ids.shape[0]
                loss, logits = self.model(input_ids, attention_mask=input_mask, labels=label_ids)
                eval_loss += loss.item()
                logits = logits.cpu().numpy()
                label_ids = label_ids.cpu().numpy()
                eval_accuracy += accuracy(logits, label_ids)

        eval_loss /= batches_count
        eval_accuracy /= data_count

        print('evaluation loss: {:.4}, accuracy: {:.4}%'.format(eval_loss, eval_accuracy * 100))

    def save_model(self, file):
        torch.save(self.model.state_dict(), file)


if __name__ == '__main__':
    train_file = 'data/small_train.csv'
    test_file = 'data/small_test.csv'
    save_file = 'saved_model/bert_seq_tuned.m'
    check_path('saved_model')
    ckpt_file = 'saved_model/bert_ckpt.m'
    num_epoch = 5
    batch_size = 8
    log_interval = 100
    twitter_sentiment = TwitterSentiment(train_file, test_file, num_epoch=num_epoch,
                                         batch_size=batch_size, log_interval=log_interval, prototype=False)
    for e in range(num_epoch):
        twitter_sentiment.train_epoch(e+1)
        twitter_sentiment.test()
        twitter_sentiment.save_model(ckpt_file)
    twitter_sentiment.save_model()