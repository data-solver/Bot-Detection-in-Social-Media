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

def sentiment_scores(self, tweet, analyser):
    """
    computes sentiment score of input tweet
    tweet - tweet which we compute sentiment score of
    analyser - sentiment analyser used to compute score
    """
    try:
        return self.analyser.polarity_scores(tweet)
    except AttributeError:
        return None
def concatenate_account_tweets(csvreader):
    """
    concatenates all the tweets of an account into one string
    csvreader - a csv reader which yields rows of our data
    """
    
    
if __name__ == '__main__':
    analyser = vader.SentimentIntensityAnalyzer()
    with open()
    