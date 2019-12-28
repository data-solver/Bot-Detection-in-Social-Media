# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 19:56:20 2019

@author: Kumar
"""
import pandas as pd
import os

if __name__ == '__main__':
    data_dir_list = []
    original_data_dir = "./Datasets/LSTM paper data/Raw Data/"
    entry = [original_data_dir, 'social_spambots_1.csv/']
    data_dir_list.append(entry)
    entry = [original_data_dir, 'social_spambots_2.csv/']
    data_dir_list.append(entry)
    entry = [original_data_dir, 'social_spambots_3.csv/']
    data_dir_list.append(entry)
    entry = [original_data_dir, 'genuine_accounts.csv/']
    data_dir_list.append(entry)
    # clean tweet datasets first
    # genuine accounts csv file is missing headers
    header = ['id', 'text', 'source', 'user_id', 'truncated', 
              'in_reply_to_status_id', 'in_reply_to_user_id',
              'in_reply_to_screen_name', 'retweeted_status_id', 'geo', 'place',
              'contributors', 'retweet_count', 'reply_count', 'favorite_count',
              'favorited', 'retweeted', 'possibly_sensitive', 'num_hashtags',
              'num_urls', 'num_mentions', 'created_at', 'timestamp',
              'crawled_at', 'updated']
    header.append('redundant')
    for index, entry in enumerate(data_dir_list):
        print('index =', index+1, 'out of', len(data_dir_list))
        with open(os.path.join(entry[0], entry[1], 'tweets.csv'), 'r',
                  encoding="Latin-1") as r:
            if 'genuine' in entry[0]:
                df = pd.read_csv(r, low_memory=False, error_bad_lines=False,
                                 names=header)
            else:
                df = pd.read_csv(r, low_memory=False, error_bad_lines=False)
        # drop NA values in relevant columns of tweets csv files
        df = df.dropna(subset=['text', 'user_id', 'retweet_count',
                               'reply_count', 'favorite_count', 'num_hashtags', 
                               'num_urls', 'num_mentions'])
        # sort by user_id column
        df = df.sort_values(['user_id'])

        df.to_csv(os.path.join('./Datasets/LSTM paper data/Clean Data/',
                               entry[1], 'tweets.csv'), index=False,
                  encoding="Latin-1")
        # clean accounts csv file
        # fill in NA values for the sparse columns
        with open(os.path.join(entry[0], entry[1], 'users.csv'), 'r',
                  encoding="Latin-1") as r:
            accounts = pd.read_csv(r)
        accounts[['geo_enabled', 'verified', 'protected',
                  'default_profile']] = \
        accounts[['geo_enabled', 'verified', 'protected',
                  'default_profile']].fillna(0)
        # drop NA values in remaining relevant columns of accounts csv files
        cols = ['statuses_count', 'followers_count', 'friends_count',
                'favourites_count', 'listed_count', 'default_profile', 
                'geo_enabled', 'profile_use_background_image', 'verified',
                'protected']
        accounts = accounts.dropna(subset=cols)
        
        # write final dataframe to csv file
        accounts.to_csv(os.path.join('./Datasets/LSTM paper data/Clean Data/',
                                     entry[1], 'users.csv'), index=False,
                        encoding="Latin-1")
