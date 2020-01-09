# -*- coding: utf-8 -*-
import pandas as pd
import os
from nltk.sentiment import vader
import matplotlib.pyplot as plt


class sentiment_analysis:
    """
    Perform sentiment analysis per tweet in our dataset and plot the resulting
    distributions of sentiment in a histogram
    """
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
            return None
    
    def avg_sentiment(self, genuine):
        if genuine:
            data = self.genuine_tweets
        else:
            data = self.bot_tweets
        score_list = []
        for tweet in data['text']:
            score = self.sentiment_scores(tweet)
            if score:
                continue
            score_list.append(score['compound'])
        return score_list


if __name__ == '__main__':
    original_data_dir = ("./Datasets/LSTM paper data")
    num_tweets = 100000000
    test = sentiment_analysis()
    test.get_data(original_data_dir, num_tweets)
    gen_sent = test.avg_sentiment(genuine=True)
    bot_sent = test.avg_sentiment(genuine=False)
    # make figure
    plt.figure()
    plt.hist(gen_sent, bins=30, label='genuine tweets sentiment',
             density=True)
    plt.hist(bot_sent, bins=30, label='bot tweets sentiment',
             density=True)
    plt.ylim((0, 1))
    plt.xlabel('sentiment score')
    plt.ylabel('proportion of tweets')
    plt.title('Histogram of sentiment score of tweets from bots and humans, '
              'tweet level')
    plt.legend(loc='upper right', bbox_to_anchor=(1.6, 0.5))
    plt.savefig(os.path.join('./Figures', 'sent_pertweet.png'))
    # do the same but per account rather than per tweet
