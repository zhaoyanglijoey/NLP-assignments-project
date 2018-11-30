import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertAdam, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader, TensorDataset
from dataloader import TweetsDataset
import pandas as pd
import torch.optim as optim
from util import convert_data_to_features, check_path
from tqdm import tqdm
import numpy as np
import time, argparse

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def positive_score(out):
    outputs = F.softmax(out, dim=1)
    score = torch.sum(outputs[:, 1])
    return score


class TwitterSentiment():
    def __init__(self, train_file, test_file, batch_size=16, num_epoch=10, log_interval=100,
                 prototype=False, parallel=False, load_model=None):
        self.log_interval = log_interval
        self.batch_size = batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.train_data = pd.read_csv(train_file, dtype={'tag':int, 'cleaned_tweet':str})
        self.test_data = pd.read_csv(test_file, dtype={'tag':int, 'cleaned_tweet':str})
        if prototype:
            self.train_data = self.train_data[:10000]
            self.test_data = self.test_data[:1000]
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
        if parallel:
            self.model = DataParallel(self.model)
        if load_model:
            self.model.load_state_dict(torch.load(load_model))

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

    def train_epoch(self, epoch, save_interval, ckpt_file):
        self.model.train()
        running_ls = 0
        start = time.time()
        num_batches = len(self.train_loader)
        for i, batch  in enumerate(self.train_loader):
            (input_ids, input_mask, label_ids) = tuple(t.to(self.device) for t in batch)

            self.model.zero_grad()
            loss, _ = self.model(input_ids, attention_mask=input_mask, labels=label_ids)
            loss.backward(torch.ones_like(loss))
            running_ls += loss.mean().item()
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
            if (i+1) % save_interval == 0:
                self.save_model(ckpt_file)

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
                loss = loss.mean()
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
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--load-model', default=None)
    argparser.add_argument('-e', '--num_epoch', type=int, default=5)
    argparser.add_argument('-t', '--test', default=False, action='store_true')
    argparser.add_argument('--pt', default=False, action='store_true', help='prototype mode')
    argparser.add_argument('-b', '--batchsize', type=int, default=32)
    argparser.add_argument('--save-interval', type=int, default=500)
    args = argparser.parse_args()


    train_file = 'data/train.csv'
    test_file = 'data/test.csv'
    save_file = 'saved_model/bert_tweet_big.m'
    check_path('saved_model')
    ckpt_file = 'saved_model/bert_ckpt.m'
    log_interval = 100
    twitter_sentiment = TwitterSentiment(train_file, test_file, num_epoch=args.num_epoch, load_model=args.load_model,
                            batch_size=args.batchsize, log_interval=log_interval, prototype=args.pt, parallel=True)
    if args.test:
        twitter_sentiment.test()
    else:
        for e in range(args.num_epoch):
            twitter_sentiment.train_epoch(e+1, args.save_interval, ckpt_file)
            twitter_sentiment.save_model(ckpt_file)
            twitter_sentiment.test()
        twitter_sentiment.save_model(save_file)