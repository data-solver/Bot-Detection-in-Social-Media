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
import time
import csv




global max_length
max_length = 0
def toToken(counter = [1,1], current_data = [0], break_outer = [0]):
    parent_dir = ("C:/Users/Kumar/OneDrive - Imperial College London/Year 3/UROP/"
                  "Dataset/cresci-2017.csv/datasets_full.csv/")
    #list of directories: ss1, ss2, ss3, genuine accounts
    data_dir = []
    data_dir.append('social_spambots_1.csv/tweets.csv')
    data_dir.append('social_spambots_2.csv/tweets.csv')
    data_dir.append('social_spambots_3.csv/tweets.csv')
    data_dir.append('genuine_accounts.csv/tweets.csv')
    while True:
        if break_outer[0] == 1:
            print("done")
            break
        path = os.path.join(parent_dir, data_dir[current_data[0]])
        with open(path, encoding = "Latin-1") as csvfile:
            #remove NA entries
            datareader = csv.reader(x.replace('\0', '') for x in csvfile)
            row = next(datareader)        
            while True:
                try:
                    try:
                        #next row in dataset
                        row = next(datareader)
                    #in case for some reason any null bytes remain
                    except csv.Error:
                        print("null byte", counter[0])
                        print(row)
                        continue
                #if we have reached the end of the file, move on to next file
                except StopIteration:
                    current_data[0] += 1
                    counter[0] = 1
                    counter[1] = 1
                    print("finished file", current_data[0])
                    break
                    #if we are at the last file, break
                    if current_data[0] > 3:
                        print("all files finished")
                        break_outer[0] = 1
                        break
                try:
                    temp = ldp.tokenizer1(row[1])
                except TypeError:
                    print("TypeError", row[1])
                    continue
                #if we are at the last file, break
                except IndexError:
                    print("****Index Error*****")
                    print(row)
                    break_outer[0] = 1
                    print("all files finished")
                    break
                tokenized_tweet = ldp.refine_token(temp)

                #store the maximum length of tokens
                global max_length
                if len(tokenized_tweet) > max_length:
                    max_length = len(tokenized_tweet)
#                    print("max_length",max_length, tokenized_tweet)
                counter[0] += 1
                
                #report progress
                if counter[0] // 10000 == counter[1]:
                    print(counter[1], "lots of 10000 entries done")
                    counter[1] += 1
                yield tokenized_tweet

tokenGen = toToken()
tokenizer = Tokenizer(filters = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(tokenGen)

#save the tokenizer to disk so we don't need to recompute
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol = pickle.HIGHEST_PROTOCOL)
    
def toPadded(tokenizer, counter = [1,1], current_data = [0], break_outer = [0]):
    parent_dir = ("C:/Users/Kumar/OneDrive - Imperial College London/Year 3/UROP/"
                  "Dataset/cresci-2017.csv/datasets_full.csv/")
    #list of directories: ss1, ss2, ss3, genuine accounts
    data_dir = []
    data_dir.append('social_spambots_1.csv/tweets.csv')
    data_dir.append('social_spambots_2.csv/tweets.csv')
    data_dir.append('social_spambots_3.csv/tweets.csv')
    data_dir.append('genuine_accounts.csv/tweets.csv')
    while True:
        if break_outer[0] == 1:
            print("done")
            break
        path = os.path.join(parent_dir, data_dir[current_data[0]])
        with open(path, encoding = "Latin-1") as csvfile:
            #remove NA entries
            datareader = csv.reader(x.replace('\0', '') for x in csvfile)
            row = next(datareader)        
            while True:
                try:
                    try:
                        #next row in dataset
                        row = next(datareader)
                   #in case for some reason any null bytes remain
                    except csv.Error:
                        print("null byte", counter[0])
                        print(row)
                        continue
                #if we have reached the end of the file, move on to next file
                except StopIteration:
                    current_data[0] += 1
                    counter[0] = 1
                    counter[1] = 1
                    print("finished file", current_data[0])
                    break
                    #if we are at the last file, break
                    if current_data[0] > 3:
                        print("all files finished")
                        break_outer[0] = 1
                        break
                try:
                    sequence = tokenizer.texts_to_sequences(row[1])
                except TypeError:
                    print("TypeError", row[1])
                    continue
                #if we are at the last file, break
                except IndexError:
                    print("****Index Error*****")
                    print(row)
                    break_outer[0] = 1
                    print("all files finished")
                    break
                temp = pad_sequences(sequence,
                                          maxlen = max_length,
                                          dtype = 'int32',
                                          padding = 'post'
                                          )    
                sequence_padded = []
                for element in temp:
                    try:
                        sequence_padded.append(element[0])
                    except IndexError:
                        continue
                    
                #if we are in the first 3 files, the tweet is from a bot
                if current_data[0] != 3:
                    label = 1
                #retweet_count, reply_count, favorite_count, num_hashtags, 
                #num_urls, num_mentions
                aux_input = [row[12], row[13], row[14], row[18], row[19], row[20]]
                
                #output has the form:
                #   padded_tweet, retweet_count, reply_count, favorite_count
                #   num_hashtags, num_urls, num_mentions, label 
                output = [sequence_padded] + aux_input + [label]
                
                counter[0] += 1
                
                #report progress
                if counter[0] // 10000 == counter[1]:
                    print(counter[1], "lots of 10000 entries done")
                    counter[1] += 1
                
                yield output

#write processed data to disk  
with open('processed_data.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    genPadded = toPadded(tokenizer)
    #write the header
    header = ['padded_tweet', 'retweet_count', 'reply_count', 'favorite_count',
              'num_hashtags', 'num_urls', 'num_mentions', 'label']
    writer.writerow(header)
    while True:
        writer.writerow(next(genPadded))
        

    
    