import os, argparse, glob2
from pytorch_pretrained_bert import BertForSequenceClassification, BertTokenizer
import torch
from clean_data import clean_tweet
import util
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.functional import softmax
import numpy as np

def subDirPath (d):
    return list(filter(os.path.isdir, [os.path.join(d,f) for f in os.listdir(d)]))

def create_dataset(tweets, tokenizer, max_seq_len):
    tweets = [clean_tweet(tweet) for tweet in tweets]
    tweets = [tweet for tweet in tweets if len(tweet) > 1]
    ids = []
    masks = []
    for tweet in tweets:
        id, mask = util.convert_to_bert_ids(tweet, tokenizer, max_seq_len)
        ids.append(id)
        masks.append(mask)
    ids = torch.stack(ids, dim=0)
    masks = torch.stack(masks, dim=0)
    tweet_dataset = TensorDataset(ids, masks)

    return tweet_dataset


def test(dataset, batchsize, model, device):
    dataloader = DataLoader(dataset, batch_size=batchsize)
    num_data = len(dataset)
    probs = np.zeros(3)
    with torch.no_grad():
        for id, mask in tqdm(dataloader):
            id = id.to(device)
            mask = mask.to(device)
            logits = model(id, attention_mask=mask)
            prob = softmax(logits, dim=-1).sum(dim=0).detach().cpu().numpy()
            # logits = logits.cpu().numpy()
            probs += prob
        probs /= num_data
        score = probs[2] - probs[0]
    return score


if __name__ == '__main__':
    data_dirs = subDirPath('testdata/')
    # print(data_dirs)
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--num-labels', type=int, default=3)
    argparser.add_argument('--load')
    argparser.add_argument('-b', '--batchsize', type=int, default=32)

    args = argparser.parse_args()

    print('creating model')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=args.num_labels)
    model = torch.nn.DataParallel(model)
    # model.load_state_dict(torch.load(args.load))
    model.to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    for data_dir in data_dirs:
        month_tweets_files = sorted(glob2.glob(os.path.join(data_dir, '*')))
        for month_tweets_file in month_tweets_files:
            print('testing', month_tweets_file)
            with open(month_tweets_file, 'r') as f:
                tweets = f.readlines()
            tweet_dataset = create_dataset(tweets, tokenizer, max_seq_len=200)
            score = test(tweet_dataset, args.batchsize, model, device)
            print(score)

