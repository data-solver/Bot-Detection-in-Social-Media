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
        with open(os.path.join(original_data_dir, "social_spambots_2.csv/tweets.csv"),
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
        counter=0
        num = 0
        if genuine:
            data = self.genuine_tweets
        else:
            data = self.bot_tweets
        score_list = []
        for tweet in data['text']:
            counter+=1
            score = self.sentiment_scores(tweet)
            score_list.append(score['compound'])
            if counter %  100000 == 0:
                num+=1
                print(num, 'lots of 100,000 entries done')
        return score_list


if __name__ == '__main__':
    original_data_dir = ("./Datasets/LSTM paper data/Clean Data")
    num_tweets = 100000000
    test = sentiment_analysis()
    test.get_data(original_data_dir, num_tweets)
    gen_sent = test.avg_sentiment(genuine=True)
    print('human done')
    bot_sent = test.avg_sentiment(genuine=False)
    # make figure
    plt.figure()
    plt.hist(gen_sent, bins=30, label='genuine tweets sentiment',
             density=True)
    plt.hist(bot_sent, bins=30, label='bot tweets sentiment',
             density=True)
    plt.ylim((0, 2))
    plt.xlabel('sentiment score')
    plt.ylabel('proportion of tweets')
    plt.title('Sentiment scores of individual tweets')
    plt.legend(loc='upper right', bbox_to_anchor=(1.6, 0.5))
    plt.savefig(os.path.join('./Figures', 'sent_pertweet.png'))
