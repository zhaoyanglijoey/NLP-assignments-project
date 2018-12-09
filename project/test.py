import os, argparse, glob2, pickle
from pytorch_pretrained_bert import BertForSequenceClassification, BertTokenizer
import torch
from clean_data import clean_tweet
import util
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.functional import softmax
import numpy as np
import matplotlib.pyplot as plt
import json

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
    positives = 0
    negatives = 0
    with torch.no_grad():
        for id, mask in tqdm(dataloader):
            id = id.to(device)
            mask = mask.to(device)
            logits = model(id, attention_mask=mask)
            # pred = torch.argmax(logits, dim=-1)
            # pred  = pred.detach().cpu().numpy()
            # positives += np.sum(pred==1)
            # negatives += np.sum(pred==0)
            prob = softmax(logits, dim=-1).sum(dim=0).detach().cpu().numpy()
            probs += prob
        probs /= num_data
        score = probs[-1] - probs[0]
        # score = (positives - negatives) / num_data
    return score



if __name__ == '__main__':
    data_dirs = subDirPath('get_tweets/output/')
    # print(data_dirs)
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--num-labels', type=int, default=3)
    argparser.add_argument('--load')
    argparser.add_argument('-b', '--batchsize', type=int, default=128)
    argparser.add_argument('-s', '--save', default='test_result_week.pkl')

    args = argparser.parse_args()

    print('creating model')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=args.num_labels)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(args.load))
    model.to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    score_dict = {}

    testoutput = 'testoutput'
    if not os.path.exists(testoutput):
        os.mkdir(testoutput)

    for data_dir in data_dirs:
        month_tweets_files = sorted(glob2.glob(os.path.join(data_dir, '*')))
        scores = {}
        account = os.path.basename(data_dir)
        # if account != 'kevinspacey':
        #     continue
        for month_tweets_file in month_tweets_files:
            print('testing', month_tweets_file)
            time_str = os.path.basename(month_tweets_file).split('_')[0]
            with open(month_tweets_file, 'r') as f:
                tweets = f.readlines()
            tweet_dataset = create_dataset(tweets, tokenizer, max_seq_len=200)
            score = test(tweet_dataset, args.batchsize, model, device)
            print(score)
            scores[time_str] = score
        plt.figure()
        plt.plot(scores.keys(), scores.values())
        plt.xlabel('time', fontsize = 16)
        plt.ylabel('score', fontsize = 16)
        plt.title('score for {}'.format(account), fontsize = 16)
        plt.ylim(-1, 1)
        plt.savefig(os.path.join(testoutput, account+'.jpg'))
        score_dict[account] = scores
        with open(os.path.join(testoutput,'{}.json'.format(account)), 'w') as f:
            json.dump(scores, f, indent=2)

    with open(os.path.join(testoutput, args.save), 'wb') as f:
        pickle.dump(score_dict, f)
    plt.show()


