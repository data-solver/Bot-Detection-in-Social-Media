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
import os
import matplotlib.pyplot as plt
import time
import pickle
from nltk.sentiment import vader


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
         user_id - fixed
         need to create barplot with percentages of sentiment distributions for
         bots and humans
"""


def conc_acc_tweets(original_data_dir, analyser, tweetno=None,
                    max_accounts=None):
    """
    concatenates all the tweets of an account into one string
    original_data_dir - data directory of tweets
    analyser - object used to perform sentiment analysis
    tweetno - maximum number of tweets per account
    max_accounts - optional limit on number of accounts to process
    """
    # genuine accounts csv file is missing headers
    header = ['id', 'text', 'source', 'user_id', 'truncated',
              'in_reply_to_status_id', 'in_reply_to_user_id',
              'in_reply_to_screen_name', 'retweeted_status_id', 'geo', 'place',
              'contributors', 'retweet_count', 'reply_count', 'favorite_count',
              'favorited', 'retweeted', 'possibly_sensitive', 'num_hashtags',
              'num_urls', 'num_mentions', 'created_at', 'timestamp',
              'crawled_at', 'updated']
    header.append('redundant')
    with open(original_data_dir, 'r', encoding="Latin-1") as r:
        if 'genuine' in original_data_dir:
            df = pd.read_csv(r, low_memory=False, error_bad_lines=False,
                             names=header)
        else:
            df = pd.read_csv(r, low_memory=False, error_bad_lines=False)
    # sort by user_id column
    df = df.sort_values(['user_id'])
    tweet_list = []
    account_sentiment_scores = []
    account_num = 0
    row_num = 0
    tweet_count = 0
    old_user_id = df['user_id'].iloc[row_num]
    length = len(df)
    # while loop till we reach end of dataframe
    while row_num < length:
        current_user_id = df['user_id'].iloc[row_num]
        # if tweet is from same user, append to list
        if current_user_id == old_user_id:
            tweet_list.append(df['text'].iloc[row_num])
            if tweetno:
                tweet_count += 1
                # if we have reached tweetno tweets, start new list
                if tweet_count == tweetno:
                    tweet = ''.join(tweet_list)
                    sent_score = sentiment_scores(tweet, analyser)
                    account_sentiment_scores.append(sent_score)
                    old_user_id = current_user_id
                    tweet_list = []
                    tweet_list.append(df['text'].iloc[row_num])
                    tweet_count = 1
        else:
            # if we have reached a new account, start new list
            tweet = ''.join(tweet_list)
            sent_score = sentiment_scores(tweet, analyser)
            account_sentiment_scores.append(sent_score)
            old_user_id = current_user_id
            tweet_list = []
            tweet_list.append(df['text'].iloc[row_num])
            account_num += 1
            if max_accounts == account_num:
                break
            if tweetno:
                tweet_count = 1
        row_num += 1
    return account_sentiment_scores

def run_model(max_accounts=None):
    analyser = vader.SentimentIntensityAnalyzer()
    # compute sentiment distributions using different numbers of maximum tweets
    tweetno_list = [10, 100, 1000, None]
    gen_sent_list = []
    bot_sent_list = []
    t1 = time.time()
    for tweetno in tweetno_list:
        print('tweetno =', tweetno)
        original_data_dir = ("./Datasets/LSTM paper data/social_spambots_1.csv/"
                             "tweets.csv")
        spambot1_sent = conc_acc_tweets(original_data_dir, analyser,
                                        tweetno=tweetno,
                                        max_accounts=max_accounts)
        print('spambot1 done')
        original_data_dir = ("./Datasets/LSTM paper data/social_spambots_2.csv/"
                             "tweets.csv")
        spambot2_sent = conc_acc_tweets(original_data_dir, analyser,
                                        tweetno=tweetno,
                                        max_accounts=max_accounts)
        print('spambot2 done')
        original_data_dir = ("./Datasets/LSTM paper data/social_spambots_3.csv/"
                             "tweets.csv")
        spambot3_sent = conc_acc_tweets(original_data_dir, analyser,
                                        tweetno=tweetno,
                                        max_accounts=max_accounts)
        print('spambot3 done')
        original_data_dir = ("./Datasets/LSTM paper data/genuine_accounts.csv/"
                             "tweets.csv")
        genuine_sent = conc_acc_tweets(original_data_dir, analyser,
                                       tweetno=tweetno,
                                       max_accounts=max_accounts)
        print('human done')
        bot_sent = spambot1_sent + spambot2_sent + spambot3_sent
        gen_sent_list.append(genuine_sent)
        bot_sent_list.append(bot_sent)
    # save lists to disk
    with open('./Sentiment Analysis/genuine_sent', 'wb') as fp:
        pickle.dump(gen_sent_list, fp)
    with open('./Sentiment Analysis/bot_sent', 'wb') as fp:
        pickle.dump(bot_sent_list, fp)
    t2 = time.time()
    print('time taken is', t2-t1)
    # plot the sentiment distribution for different tweet numbers
    length = len(bot_sent_list)
    for i in range(length):
        genuine_sent = gen_sent_list[i]
        bot_sent = bot_sent_list[i]
        if tweetno_list[i]:
            savename = "sent_dist%i.png" % tweetno_list[i]
        else:
            savename = "sent_dist.png"
        plt.figure()
        plt.hist(genuine_sent, bins=30, label='genuine tweets sentiment',
                 density=True)
        plt.hist(bot_sent, bins=30, label='bot tweets sentiment',
                 density=True)
        plt.ylim((0, 1))
        plt.xlabel('sentiment score')
        plt.ylabel('proportion of tweets')
        if tweetno_list[i]:
            plt.title('Sentiments, account level, tweetno = %i' % tweetno_list[i])
        else:
            plt.title('Sentiments, account level, no tweet limit')
        plt.legend(loc='upper right', bbox_to_anchor=(1.6, 0.5))
        plt.savefig(os.path.join('./Figures', savename))
