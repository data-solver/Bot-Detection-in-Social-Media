# -*- coding: utf-8 -*-

"""
idea:
    create an empty csv file 'data'
    
    create a generator function which:
        takes a tweet from a line in the original 4 csv files, tokenizes the tweet using
        our ldp.refine_token function and yields the result * make sure generator stops
        once full data is covered, as an aside: store the maximum length of the tokenized tweets
        across all generator calls - max_length
    
    fit a keras.preprocessing.text.Tokenizer on this generator 
    
    create a function which goes line by line in the original 4 csv files and:
        transform the tweet to a sequence
        pads the sequence to max_length
        writes each sequence to a new row under the column 'text' in our 'data' csv file
    
    go line by line and write the 6 columns of auxilliary input to our 'data' csv file
    
    
    create embedding matrix for words in our training set
    
    write the list of padded sequences to a csv file, include labels for bot or not
    and include 6 columns for the auxilliary input
    
    final csv file should have 8 columns: 1 for padded sequence representation
    of tweet, 6 for auxilliary input, 1 for label (bot or not)
    
    then pass
"""
import lstm_data_processing as ldp
import pandas as pd
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

global max_length
max_length = 0
def toToken(counter = [1,1], current_data = [0]):
    parent_dir = ("C:/Users/Kumar/OneDrive - Imperial College London/Year 3/UROP/"
                  "Dataset/cresci-2017.csv/datasets_full.csv/")
    #list of directories: ss1, ss2, ss3, genuine accounts
    data_dir = []
    data_dir.append('social_spambots_1.csv/tweets.csv')
    data_dir.append('social_spambots_2.csv/tweets.csv')
    data_dir.append('social_spambots_3.csv/tweets.csv')
    data_dir.append('genuine_accounts.csv/tweets.csv')

    while True:
        path = os.path.join(parent_dir, data_dir[current_data[0]])
        try:
            row = pd.read_csv(path, skiprows = range(1, counter[0]), nrows = 1,
                              encoding = "Latin-1")
            if row.isna()['text'][0] == True:
                counter[0] += 1
                continue
        except StopIteration:
            current_data[0] += 1
            print("finished file", current_data[0])
            if current_data[0] > 3:
                print("all files finished")
                break
        temp = ldp.tokenizer1(row['text'][0])
        tokenized_tweet = ldp.refine_token(temp)
        
        #store the maximum length
        global max_length
        if len(tokenized_tweet) > max_length:
            max_length = len(tokenized_tweet)
        counter[0] += 1
        if counter[0] // 1000 == counter[1]:
            counter[1] += 1
            print(counter[1], "lots of 1000 entries done")
        yield tokenized_tweet

tokenGen = toToken()
tokenizer = Tokenizer(filters = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(tokenGen)

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol = pickle.HIGHEST_PROTOCOL)