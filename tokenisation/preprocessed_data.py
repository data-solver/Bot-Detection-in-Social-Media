# -*- coding: utf-8 -*-
import lstm_data_processing as ldp
import pandas as pd
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import csv

# directories for files and data
# original data
original_data_dir = ("C:/Users/Kumar/OneDrive - Imperial College London/Year 3"
                     "/UROP/Dataset/cresci-2017.csv/datasets_full.csv/")
# tokenizer fit on training set
tokenizer_dir = ("C:/Users/Kumar/OneDrive - Imperial College London/"
                 "Github repositories/Bot-Detection-in-Social-Media/"
                 "tokenisation")
# pre-processed data
proc_data_dir = ("C:/Users/Kumar/OneDrive - Imperial College London/Year 3/"
                 "UROP/Dataset")


def toToken(counter=[1, 1], current_data=[0], break_outer=[0],
            parent_dir=original_data_dir):
    # list of directories: ss1, ss2, ss3, genuine accounts
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
        with open(path, encoding="Latin-1") as csvfile:
            # remove NA entries
            datareader = csv.reader(x.replace('\0', '') for x in csvfile)
            row = next(datareader)
            while True:
                try:
                    try:
                        # next row in dataset
                        row = next(datareader)
                    # in case for some reason any null bytes remain
                    except csv.Error:
                        print("null byte", counter[0])
                        print(row)
                        continue
                # if we have reached the end of the file, move on to next file
                except StopIteration:
                    current_data[0] += 1
                    counter[0] = 1
                    counter[1] = 1
                    print("finished file", current_data[0])
                    break
                # if we are at the last file, break
                if current_data[0] > 3:
                    print("all files finished")
                    break_outer[0] = 1
                    break
                try:
                    temp = ldp.tokenizer1(row[1])
                except TypeError:
                    print("TypeError", row[1])
                    continue
                # if we are at the last file, break
                except IndexError:
                    print("****Index Error*****")
                    print(row)
                    break_outer[0] = 1
                    print("all files finished")
                    break
                tokenized_tweet = ldp.refine_token(temp)
                counter[0] += 1
                # report progress
                if counter[0] // 10000 == counter[1]:
                    print(counter[1], "lots of 10000 entries done")
                    counter[1] += 1
                yield tokenized_tweet


def toPadded(tokenizer, counter=[1, 1], current_data=[0], break_outer=[0],
             parent_dir=original_data_dir):
    # list of directories: ss1, ss2, ss3, genuine accounts
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
        with open(path, encoding="Latin-1") as csvfile:
            # remove NA entries
            datareader = csv.reader(x.replace('\0', '') for x in csvfile)
            row = next(datareader)
            while True:
                try:
                    try:
                        # next row in dataset
                        row = next(datareader)
                    # in case for some reason any null bytes remain
                    except csv.Error:
                        print("null byte", counter[0])
                        print(row)
                        continue
                # if we have reached the end of the file, move on to next file
                except StopIteration:
                    current_data[0] += 1
                    counter[0] = 1
                    counter[1] = 1
                    print("finished file", current_data[0])
                    break
                # if we are at the last file, break
                if current_data[0] > 3:
                    print("all files finished")
                    break_outer[0] = 1
                    break
                try:
                    sequence = tokenizer.texts_to_sequences([row[1]])
                except TypeError:
                    print("TypeError", row[1])
                    continue
                # if we are at the last file, break
                except IndexError:
                    print("****Index Error*****")
                    print(row)
                    break_outer[0] = 1
                    print("all files finished")
                    break
                sequence_padded = pad_sequences(sequence,
                                                maxlen=max_length,
                                                dtype='int32',
                                                padding='post'
                                                )[0]
                # change sequence_padded to a list
                sequence_padded = sequence_padded.tolist()
                # if we are in the first 3 files, the tweet is from a bot
                if current_data[0] != 3:
                    label = 1
                # tweets from the last file are from humans
                elif current_data[0] == 3:
                    label = 0
                # retweet_count, reply_count, favorite_count, num_hashtags,
                # num_urls, num_mentions
                aux_input = [row[12], row[13], row[14], row[18], row[19],
                             row[20]]
                # output has the form:
                #   padded_tweet, retweet_count, reply_count, favorite_count
                #   num_hashtags, num_urls, num_mentions, label
                output = [sequence_padded] + aux_input + [label]
                counter[0] += 1
                # report progress
                if counter[0] // 10000 == counter[1]:
                    print(counter[1], "lots of 10000 entries done")
                    counter[1] += 1
                yield output


if __name__ == '__main__':
    # maximum length to pad/truncate tokenized tweets to
    max_length = 30
    # whether or not to work with reduced dataset
    shrink_data = False
    # amount of data to work with
    length = 2000000
    # number of words for tokenizer
    num_words = 30000
    # create tokenizer
    tokenGen = toToken()
    tokenizer = Tokenizer(num_words=num_words,
                          filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(tokenGen)
    # save the tokenizer to disk so we don't need to recompute
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # load tokenizer
    with open(os.path.join(tokenizer_dir, 'tokenizer.pickle'), 'rb') as handle:
        tokenizer = pickle.load(handle)
    # write processed data to disk
    with open(os.path.join(proc_data_dir, 'processed_data.csv'), 'w',
              newline='') as csvfile:
        writer = csv.writer(csvfile)
        genPadded = toPadded(tokenizer)
        # write the header
        header = ['padded_tweet', 'retweet_count', 'reply_count',
                  'favorite_count', 'num_hashtags', 'num_urls', 'num_mentions',
                  'label']
        writer.writerow(header)
        while True:
            try:
                writer.writerow(next(genPadded))
            # stop when we reach the end of the file
            except StopIteration:
                break
    # shuffle the rows of this processed_data and write the shuffled version to
    # a new file
    with open(os.path.join(proc_data_dir, 'processed_data.csv'), 'r') as r, \
        open(os.path.join(proc_data_dir, 'shuffled_processed_data.csv'), 'w',
             newline='') as w:
        writer = csv.writer(w)
        # write the header
        header = ['padded_tweet', 'retweet_count', 'reply_count',
                  'favorite_count', 'num_hashtags', 'num_urls', 'num_mentions',
                  'label']
        writer.writerow(header)
        # load processed_data into dataframe (its small enough to fit in RAM)
        df = pd.read_csv(r).to_numpy()
        # shuffle the rows of this csv file
        size = len(df)
        indices_list = np.arange(1, size)
        indices_shuffled = np.random.choice(indices_list,
                                            size=len(indices_list),
                                            replace=False)
        counter = 0
        for element in indices_shuffled:
            writer.writerow(df[element])
            counter += 1
            if shrink_data:
                if counter == length:
                    break
