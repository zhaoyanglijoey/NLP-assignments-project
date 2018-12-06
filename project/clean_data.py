import pandas as pd
import os
from sklearn.model_selection import train_test_split
import html
from pytorch_pretrained_bert import BasicTokenizer

basic_tokenizer = BasicTokenizer()

def clean_tweet(tweet):
    tweet = html.unescape(tweet)
    words = tweet.split()
    words = ' '.join([word.lower() for word in words
                      if not (word.startswith('@') or word.startswith('http') or '/' in word)])
    words = ''.join([ c if c.isalnum() or c in '\'\"?!' else ' ' for c in words ])
    return words

def check_invalid(tweet):
    tweet = html.unescape(tweet)
    words = tweet.split()
    for word in words:
        try:
            word.encode('ascii')
        except UnicodeEncodeError:
            return 1
    return 0


if __name__ == '__main__':
    path = os.path.join('data', 'tweets.csv')
    df = pd.read_csv(path, encoding='latin1', usecols=[0, 5], header=None, names=['tag', 'tweet'])
    df['invalid'] = df.tweet.apply(check_invalid)
    cleandf = df[df.invalid == 0]
    cleandf['cleaned_tweet'] = cleandf.tweet.apply(clean_tweet)

    cleandf = cleandf[cleandf.cleaned_tweet.apply(len) > 1]
    cleandf = cleandf.reset_index(drop=True)
    cleandf['tag'][cleandf['tag'] == 4] = 1

    # small_df, _ = train_test_split(cleandf, test_size=0.8)
    # small_train, small_test = train_test_split(small_df, test_size=0.1)
    # small_train = small_train.reset_index(drop=True)
    # small_test = small_test.reset_index(drop=True)
    # small_train.to_csv('data/small_train.csv', index=False, columns=['tag', 'cleaned_tweet'])
    # small_test.to_csv('data/small_test.csv', index=False, columns=['tag', 'cleaned_tweet'])

    train, test = train_test_split(cleandf, test_size=0.1, random_state=42)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    train.to_csv('data/train.csv', index=False, columns=['tag', 'cleaned_tweet'])
    test.to_csv('data/test.csv', index=False, columns=['tag', 'cleaned_tweet'])