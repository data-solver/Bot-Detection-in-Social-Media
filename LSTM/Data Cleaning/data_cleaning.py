# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 19:56:20 2019

@author: Kumar
"""
import pandas as pd
import os
import csv

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
    # first we fix the genuine accounts csv and rewrite it (it has bad rows)
    entry = data_dir_list[3]
    with open(os.path.join(entry[0], 'genuine_accounts_old.csv', 'tweets.csv'),
              'r', encoding='Latin-1') as r:
        reader = csv.reader(r)
        rows_list = []
        # define the criteria for checking if row is usable (bad csv file)
        criteria = ['\\N', '\\N', '', '']
        for row in reader:
            # only keep usable rows
            if row[9:13] == criteria:
                row.pop(9)
                dict1 = {}
                # key = column name - get row in dict format
                for index, s in enumerate(row):
                    dict1[header[index]] = s
                rows_list.append(dict1)
            else:
                continue
        # create dataframe from good rows
        df = pd.DataFrame(rows_list, columns=header)
        # save to csv file
        df.to_csv(os.path.join('./Datasets/LSTM paper data/Raw Data/',
                               entry[1], 'tweets.csv'), index=False,
                  encoding="Latin-1")
    # proceed with data cleaning
    for index, entry in enumerate(data_dir_list):
        print('index =', index+1, 'out of', len(data_dir_list))
        with open(os.path.join(entry[0], entry[1], 'tweets.csv'), 'r',
                  encoding="Latin-1") as r:
            if 'genuine' in entry[1]:
                df = pd.read_csv(r, low_memory=False,
                                 names=header)
            else:
                df = pd.read_csv(r, low_memory=False)
        # list of columns we will use for analysis
        # we exclude 'retweet_count' due to high number of NA values
        relevant_columns = ['text', 'user_id', 'retweet_count',
                            'reply_count', 'favorite_count', 'num_hashtags',
                            'num_urls', 'num_mentions']
        # remove columns we don't need
        df = df[relevant_columns]
        df['retweet_count'].fillna(0)
        # drop NA values in relevant columns of tweets csv files
        df.dropna(subset=relevant_columns, inplace=True)
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
        # list of relevant columns in users csv file
        cols = ['statuses_count', 'followers_count', 'friends_count',
                'favourites_count', 'listed_count', 'default_profile',
                'geo_enabled', 'profile_use_background_image', 'verified',
                'protected']
        # remove columns we don't need
        accounts = accounts[cols]
        # drop NA values in remaining relevant columns of accounts csv files
        accounts = accounts.dropna(subset=cols)
        # write final dataframe to csv file
        accounts.to_csv(os.path.join('./Datasets/LSTM paper data/Clean Data/',
                                     entry[1], 'users.csv'), index=False,
                        encoding="Latin-1")
