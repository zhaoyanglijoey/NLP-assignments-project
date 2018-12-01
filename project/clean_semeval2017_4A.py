import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np
import html

import clean_data


if __name__ == '__main__':
    folder = os.path.join('datastories-semeval2017-task4')
    path = os.path.join(folder, 'train.txt')
    df = pd.read_csv(path, encoding='latin1', usecols=[1, 2], header=None, names=['tag', 'tweet'],  delimiter='\t')
    df['cleaned_tweet'] = df.tweet.apply(clean_data.clean_tweet)
    df['invalid'] = df.tweet.apply(clean_data.check_invalid)
    cleandf = df[df.invalid == 0]
    cleandf = cleandf.reset_index(drop=True)
    # cleandf['tag'][cleandf['tag'] >= 1] = 1
    cleandf['tag'][cleandf['tag'] == 'negative'] = 0
    cleandf['tag'][cleandf['tag'] == 'neutral'] = 1
    cleandf['tag'][cleandf['tag'] == 'positive'] = 2
    # small_df, _ = train_test_split(cleandf, test_size=0.8)
    small_df = cleandf
    small_train, small_test = train_test_split(small_df, test_size=0.1)
    small_train = small_train.reset_index(drop=True)
    small_test = small_test.reset_index(drop=True)
    small_train.to_csv(os.path.join(folder, 'small_train.csv'), index=False, columns=['tag', 'cleaned_tweet'])
    small_test.to_csv(os.path.join(folder, 'small_test.csv'), index=False, columns=['tag', 'cleaned_tweet'])

    # train, test = train_test_split(cleandf, test_size=0.1)
    # train = train.reset_index(drop=True)
    # test = test.reset_index(drop=True)
    # train.to_csv('data/train.csv', index=False, columns=['tag', 'cleaned_tweet'])
    # test.to_csv('data/test.csv', index=False, columns=['tag', 'cleaned_tweet'])