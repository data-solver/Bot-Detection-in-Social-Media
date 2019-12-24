# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 14:05:18 2019

@author: Kumar
"""

concat_string = 'you are very bad and evil'

seperate_strings = ['you', 'are very bad', 'and evil']

analyser = vader.SentimentIntensityAnalyzer()
scores = analyser.polarity_scores(concat_string)['compound']
scores_list = []
for string in seperate_strings:
    s = analyser.polarity_scores(string)['compound']
    scores_list.append(s)
scores_list
np.mean(scores_list)