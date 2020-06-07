# -*- coding: utf-8 -*-
import pandas as pd
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from data_preprocessing import lstm_data_processing as ldp
import pickle
import numpy as np
import csv
import random


def toToken(original_data_dir, counter=[1, 1], current_data=[0],
            break_outer=[0]):
    parent_dir = original_data_dir
    # list of directories: ss1, ss2, ss3, genuine accounts
    data_dir = ['usdatav1.csv']
    path = os.path.join(parent_dir, data_dir[current_data[0]])
    with open(path, encoding="Latin-1") as csvfile:
        # remove NA entries
        datareader = csv.reader(x.replace('\0', '') for x in csvfile)
        row = next(datareader)
        t = row.index('text')
        while True:
            # if we are at the last file, break
            if current_data[0] > 3:
                print("all files finished")
                break_outer[0] = 1
                break
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
            try:
                temp = ldp.tokenizer1(row[t])
            except TypeError:
                print("TypeError", row[t])
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
            if counter[0] // 100000 == counter[1]:
                print(counter[1], "lots of 100000 entries done")
                counter[1] += 1
            yield tokenized_tweet


def toPadded(tokenizer, max_length, original_data_dir, counter=[1, 1],
             current_data=[0],
             break_outer=[0]):
    parent_dir = original_data_dir
    # list of directories: ss1, ss2, ss3, genuine accounts
    data_dir = ['usdatav1.csv']
    path = os.path.join(parent_dir, data_dir[current_data[0]])
    with open(path, encoding="Latin-1") as csvfile:
        # remove NA entries
        datareader = csv.reader(x.replace('\0', '') for x in csvfile)
        header = next(datareader)
        relevant_cols = ['retweet_count',
                         'favorite_count', 'num_hashtag',
                         'num_urls', 'num_mentions']
        indices = [header.index(relevant_cols[i]) for i in \
                   range(len(relevant_cols))]
        # get index of tweets and user id
        t_id = header.index('text')
        u_id = header.index('user_id')
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
                sequence = tokenizer.texts_to_sequences([row[t_id]])
            except TypeError:
                print("TypeError", row[t_id])
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
            # retweet_count, reply_count, favorite_count, num_hashtags,
            # num_urls, num_mentions
            aux_input = [row[indices[i]] for i in range(len(indices))]
            # output has the form:
            #   padded_tweet, user_id, retweet_count, reply_count,
            #   favorite_count, num_hashtags, num_urls, num_mentions, label
            output = [sequence_padded] + [row[u_id]] + aux_input
            counter[0] += 1
            # report progress
            if counter[0] // 100000 == counter[1]:
                print(counter[1], "lots of 100000 entries done")
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


def process_data(tokenizer, max_length, proc_data_dir,
                 original_data_dir):
    # write processed data to disk
    with open(os.path.join(proc_data_dir, 'processed_data.csv'), 'w',
              newline='') as csvfile:
        writer = csv.writer(csvfile)
        genPadded = toPadded(tokenizer, max_length, original_data_dir)
        # write the header
        header = ['padded_tweet', 'user_id', 'retweet_count',
                  'favorite_count', 'num_hashtag', 'num_urls', 'num_mentions']
        writer.writerow(header)
        while True:
            try:
                writer.writerow(next(genPadded))
            # stop when we reach the end of the file
            except StopIteration:
                break


def shuffle_data(length, proc_data_dir):
    with open(os.path.join(proc_data_dir, 'processed_data.csv'), 'r') as r:
        # load processed_data into dataframe (its small enough to fit in RAM)
        df = pd.read_csv(r)
        # shuffle the rows of this csv file
        groups = [df for _, df in df.groupby('user_id')]
        random.shuffle(groups)
        df = pd.concat(groups).reset_index(drop=True)
        path = os.path.join(proc_data_dir, 'shuffled_processed_data.csv')
        df.to_csv(path, index=False)


def run_processing(num_words, data_dirs, length=False, max_length=30,
                   new_tokenizer=True, proc_data=True, shuffle=True):
    """
    num_words - number of words for tokenizer
    data_dirs - list of directories with the following format:
        original_data_dir
        tokenizer_dir
        proc_data_dir
        glove_dir (not used here)
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
        process_data(tokenizer, max_length, data_dirs[2], data_dirs[0])
    if shuffle:
        shuffle_data(length, data_dirs[2])
    pass
