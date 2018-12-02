import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np
import html

import clean_data


def txt_to_clean_csv(filepath_txt):
    df = pd.read_csv(filepath_txt, encoding='latin1', usecols=[1, 2],
                     header=None, names=['tag', 'tweet'],  delimiter='\t',
                     quoting=3)
    df['invalid'] = df.tweet.apply(clean_data.check_invalid)
    cleandf = df[df.invalid == 0]
    cleandf['cleaned_tweet'] = cleandf.tweet.apply(clean_data.clean_tweet)
    cleandf = cleandf[cleandf.cleaned_tweet.apply(len) > 1]
    cleandf = cleandf.reset_index(drop=True)
    # cleandf['tag'][cleandf['tag'] >= 1] = 1
    cleandf['tag'][cleandf['tag'] == 'negative'] = 0
    cleandf['tag'][cleandf['tag'] == 'neutral'] = 1
    cleandf['tag'][cleandf['tag'] == 'positive'] = 2

    return cleandf


if __name__ == '__main__':
    folder = os.path.join('datastories-semeval2017-task4')
    train_path = os.path.join(folder, 'train.tsv')
    test_path = os.path.join(folder, 'test.tsv')
    train_df = txt_to_clean_csv(train_path)
    test_df = txt_to_clean_csv(test_path)
    # small_df, _ = train_test_split(cleandf, test_size=0.8)
    train_df.to_csv(os.path.join(folder, 'train.csv'), index=False, columns=['tag', 'cleaned_tweet'])
    test_df.to_csv(os.path.join(folder, 'test.csv'), index=False, columns=['tag', 'cleaned_tweet'])

    # train, test = train_test_split(cleandf, test_size=0.1)
    # train = train.reset_index(drop=True)
    # test = test.reset_index(drop=True)
    # train.to_csv('data/train.csv', index=False, columns=['tag', 'cleaned_tweet'])
    # test.to_csv('data/test.csv', index=False, columns=['tag', 'cleaned_tweet'])
