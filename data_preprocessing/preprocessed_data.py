# -*- coding: utf-8 -*-
import pandas as pd
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from data_preprocessing import lstm_data_processing as ldp
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


def toToken(original_data_dir, counter=[1, 1], current_data=[0],
            break_outer=[0]):
    parent_dir = original_data_dir
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


def toPadded(tokenizer, max_length, original_data_dir, counter=[1, 1],
             current_data=[0],
             break_outer=[0]):
    parent_dir = original_data_dir
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


def create_tokenizer(num_words, original_data_dir, tokenizer_dir):
    # create tokenizer
    tokenGen = toToken(original_data_dir)
    tokenizer = Tokenizer(num_words=num_words,
                          filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(tokenGen)
    # save the tokenizer to disk so we don't need to recompute
    with open(os.path.join(tokenizer_dir, 'tokenizer.pickle'), 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pass


def process_data(tokenizer, max_length, length, proc_data_dir,
                 original_data_dir):
    # write processed data to disk
    with open(os.path.join(proc_data_dir, 'processed_data.csv'), 'w',
              newline='') as csvfile:
        writer = csv.writer(csvfile)
        genPadded = toPadded(tokenizer, max_length, original_data_dir)
        # write the header
        header = ['padded_tweet', 'retweet_count', 'reply_count',
                  'favorite_count', 'num_hashtags', 'num_urls', 'num_mentions',
                  'label']
        writer.writerow(header)
        counter = 0
        while True:
            try:
                writer.writerow(next(genPadded))
                counter += 1
                # only take 'length' number of rows
                if length:
                    if counter == length:
                        break
            # stop when we reach the end of the file
            except StopIteration:
                break


def shuffle_data(length, proc_data_dir):
    if not length:
        return
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
            if length:
                if counter == length:
                    break


def run_processing(num_words, data_dirs, length=False, max_length=30,
                   new_tokenizer=True, proc_data=True, shuffle=True):
    """
    num_words - number of words for tokenizer
    data_dirs - list of directories with the following format:
        original_data_dir
        tokenizer_dir
        proc_data_dir
    length - amount of data to work with, if False, work with full data
    max_length - maximum length to pad/truncate tokenized tweets to
    shrink_data - whether or not to work with reduced dataset
    new_tokenizer - whether or not to create a new tokenizer or load a
                    previously generated one
    proc_data - whether or not to process the original data or load previously
                processed data.
    shuffle_data - whether or not to shuffle the processed data
    """
    if new_tokenizer:
        # create tokenizer
        create_tokenizer(num_words, data_dirs[0], data_dirs[1])
    # load tokenizer
    with open(os.path.join(data_dirs[1], 'tokenizer.pickle'), 'rb') as handle:
        tokenizer = pickle.load(handle)
    # write processed data to disk
    if proc_data:
        process_data(tokenizer, max_length, length, data_dirs[2], data_dirs[0])
    if shuffle:
        shuffle_data(length, data_dirs[2])
    pass
