# -*- coding: utf-8 -*-
"""
First we concatenate all the tweets of 1 account into one string
Perform sentiment analysis on this string
Append sentiment score to a list

Repeat for all accounts in dataset

Keep two separate lists, one for bots, one for humans

Repeat histogram plot done in sentiment analysis per tweet
"""

import pandas as pd
import csv
import os
from nltk.sentiment import vader
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle


def sentiment_scores(tweet, analyser):
    """
    computes sentiment score of input tweet
    tweet - tweet which we compute sentiment score of
    analyser - sentiment analyser used to compute score
    """
    try:
        return analyser.polarity_scores(tweet)['compound']
    except AttributeError:
        return None
    
"""
problem: entries of genuine_accounts.csv/tweets.csv is not arranged by
         user_id, so need to check 
"""
def concatenate_account_tweets(original_data_dir, analyser, max_accounts=None):
    """
    concatenates all the tweets of an account into one string
    original_data_dir - data directory of tweets
    analyser - object used to perform sentiment analysis
    max_tweets - optional limit on number of accounts to process
    """
    with open(original_data_dir, 'r', encoding="Latin-1") as r:
        reader = csv.reader(r)
        #skip headers of csv file
        headers = next(reader)
        # get indices for user_id and tweets
        user_id_index = headers.index('user_id')
        tweet_index = headers.index('text')
        tweet = ""
        row = next(reader)
        # get id of first account
        current_user_id = row[user_id_index]
        # empty list to store sentiment score of each account
        account_sentiment_scores = []
        b=[]
        # list of tweets to be concatenated together
        tweet_list = []
        # count number of accounts we have gone through
        account_num = 0
        row_num=1
        while True:
            row_num+=1
            try:
                row = next(reader)
                # skip blank rows
                if row == []:
                    continue
            # skip NA entries
            except csv.Error:
                continue
            # if we reach end of file, break
            except StopIteration:
                print("end of file at row", row_num)
                break
            # if the tweet is from the same account, append it to our 'big' 
            # tweet
            try:
                if current_user_id == row[user_id_index]:
                    tweet_list.append(row[tweet_index])
                    continue
                # if tweet is from another account, compute sentiment score 
                # and reset tweet
                else:
                    tweet = ''.join(tweet_list)
                    sent_score = sentiment_scores(tweet, analyser)
                    account_sentiment_scores.append(sent_score)
                    account_num += 1
                    if max_accounts == account_num:
                        break
                    current_user_id = row[user_id_index]
                    tweet_list = []
                    tweet_list.append(row[tweet_index])
            # dealing with last row of genuine_tweets.csv 
            except IndexError:
                tweet = ''.join(tweet_list)
                sent_score = sentiment_scores(tweet, analyser)
                account_sentiment_scores.append(sent_score)
                break
    return account_sentiment_scores
    
    
if __name__ == '__main__':
    analyser = vader.SentimentIntensityAnalyzer()
    max_accounts=None
    t1=time.time()
    original_data_dir = ("./Datasets/LSTM paper data/social_spambots_1.csv/"
                        "tweets.csv")
    spambot1_sent = concatenate_account_tweets(original_data_dir, analyser,
                                               max_accounts=max_accounts)
    original_data_dir = ("./Datasets/LSTM paper data/social_spambots_2.csv/"
                        "tweets.csv")
    spambot2_sent = concatenate_account_tweets(original_data_dir, analyser,
                                               max_accounts=max_accounts)
    original_data_dir = ("./Datasets/LSTM paper data/social_spambots_3.csv/"
                        "tweets.csv")
    spambot3_sent = concatenate_account_tweets(original_data_dir, analyser,
                                               max_accounts=max_accounts)
    
    original_data_dir = ("./Datasets/LSTM paper data/genuine_accounts.csv/"
                        "tweets.csv")
    genuine_sent = concatenate_account_tweets(original_data_dir, analyser,
                                              max_accounts=max_accounts)
    
    bot_sent = spambot1_sent + spambot2_sent + spambot3_sent 
    
    # save lists to disk
    
    with open('./Sentiment Analysis/genuine_sent', 'wb') as fp:
        pickle.dump(genuine_sent, fp)
    
    with open('./Sentiment Analysis/bot_sent', 'wb') as fp:
        pickle.dump(bot_sent, fp)
    t2 = time.time()
    print('time taken is', t2-t1)
    
    plt.figure()
    plt.hist(genuine_sent, bins=30, label='genuine tweets sentiment',
             density=True)
    plt.hist(bot_sent, bins=30, label='bot tweets sentiment',
             density=True)
    plt.ylim((0,1))
    plt.xlabel('sentiment score')
    plt.ylabel('proportion of tweets')
    plt.title('Histogram of sentiment score of tweets from bots and humans, account level')
    plt.legend(loc='upper right', bbox_to_anchor=(1.6, 0.5))
    
        
                
            
            
        
    
    