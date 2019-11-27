# -*- coding: utf-8 -*-
import pandas as pd
import csv
import os
from nltk.sentiment import vader
import numpy as np
import matplotlib.pyplot as plt

class sentiment_analysis:
    def __init__(self):
        self.analyser = vader.SentimentIntensityAnalyzer()
        pass
    def get_data(self, original_data_dir, num_tweets):
        with open(os.path.join(original_data_dir, "genuine_accounts.csv/tweets.csv"),
                               'r', encoding="Latin-1") as r:
            self.genuine_tweets = pd.read_csv(r, nrows=num_tweets//2)
        
        with open(os.path.join(original_data_dir, "social_spambots_1.csv/tweets.csv"),
                  'r', encoding="Latin-1") as r:
            self.bot_tweets = pd.read_csv(r, nrows=num_tweets//6)
        with open(os.path.join(original_data_dir, "social_spambots_1.csv/tweets.csv"),
                  'r', encoding="Latin-1") as r:
            self.bot_tweets.append(pd.read_csv(r, nrows=num_tweets//6))
        with open(os.path.join(original_data_dir, "social_spambots_3.csv/tweets.csv"),
                  'r', encoding="Latin-1") as r:
            self.bot_tweets.append(pd.read_csv(r, nrows=num_tweets//6))
        
    
    def sentiment_scores(self, tweet):
        try:
            return self.analyser.polarity_scores(tweet)
        except AttributeError:
#            print(tweet)
            return None
#        print("{:-<40} {}".format(tweet, str(score)))
    
    def avg_sentiment(self, genuine):
        if genuine == True:
            data = self.genuine_tweets
        elif genuine == False:
            data = self.bot_tweets
        score_list = []
        for tweet in data['text']:
            score = self.sentiment_scores(tweet)
            if score == None:
                continue
            score_list.append(score['compound'])
        return score_list
    
if __name__ == '__main__':
    original_data_dir = ("C:/Users/Kumar/OneDrive - Imperial College London/"
                     "Year 3/UROP/Dataset/cresci-2017.csv/"
                     "datasets_full.csv/")
    test = sentiment_analysis()
    test.get_data(original_data_dir, 100000)
    avg_gen_sent = test.avg_sentiment(genuine=True)
    avg_bot_sent = test.avg_sentiment(genuine=False)
    
    plt.figure()
    plt.hist(avg_gen_sent, bins=30, label='genuine tweets sentiment',
             density=True)
    plt.hist(avg_bot_sent, bins=30, label='bot tweets sentiment',
             density=True)
    plt.ylim((0,1))
    plt.xlabel('sentiment score')
    plt.ylabel('proportion of tweets')
    plt.title('Histogram of sentiment score of tweets from bots and humans')
    plt.legend(loc='upper right', bbox_to_anchor=(1.6, 0.5))
    
    # do the same but per account rather than per tweet
    

