import pandas as pd
import os
from sklearn.model_selection import train_test_split
import html


def valid_word(word):
    if word.startswith('@'):
        return False
    try:
        word.encode('ascii')
    except UnicodeEncodeError:
        return False

    return True


def clean_tweet(tweet):
    tweet = html.unescape(tweet)
    words = tweet.split()
    words = [word.lower() for word in words if valid_word(word)]

    return ' '.join(words)


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
    df['cleaned_tweet'] = df.tweet.apply(clean_tweet)
    df['invalid'] = df.tweet.apply(check_invalid)
    cleandf = df[df.invalid == 0]
    cleandf = cleandf.reset_index(drop=True)
    train, test = train_test_split(cleandf, test_size=0.1)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    train.to_csv('data/train.csv', index=False, columns=['tag', 'cleaned_tweet'])
    test.to_csv('data/test.csv', index=False, columns=['tag', 'cleaned_tweet'])