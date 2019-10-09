# -*- coding: utf-8 -*-

import pandas as pd
import lstm_data_processing as ldp
from tensorflow.keras.preprocessing.text import Tokenizer

#number of rows to test with
n = 100

with open(r'C:\Users\Kumar\OneDrive - Imperial College London\Year 3\UROP\\'
          r'Dataset\cresci-2017.csv\datasets_full.csv\social_spambots_2.csv'
          r'\tweets.csv',
          encoding = 'Latin-1') as f:
    df = pd.read_csv(f)
    df = df.head(n)
tokenized_tweets = []
for row in df['text']:
    temp = ldp.tokenizer1(row)
    tokenized_tweets.append(ldp.refine_token(temp))
    
tokenizer = Tokenizer(num_words = 1000)
tokenizer.fit_on_texts(tokenized_tweets)
sequences = tokenizer.texts_to_sequences(tokenized_tweets)
word_index = tokenizer.word_index