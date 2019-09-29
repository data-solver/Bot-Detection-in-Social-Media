# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
import csv
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Constant

# output dimension of lstm layer
lstm_dim = 32
# directories for files and data
# tokenizer
tokenizer_dir = ("C:/Users/Kumar/OneDrive - Imperial College London/"
                 "Github repositories/Bot-Detection-in-Social-Media/"
                 "tokenisation")
# pre-processed data
proc_data_dir = ("C:/Users/Kumar/OneDrive - Imperial College London/Year 3/"
                 "UROP/Dataset")
# glove embedding
glove_dir = ("C:/Users/Kumar/OneDrive - Imperial College London/Year 3/"
             "UROP/glove.twitter.27B")

"""
auxilliary inputs are:
    retweet count
    reply count
    favourite count
    number of hashtags
    number of urls
    number of mentions
"""


class myModel:
    def __init__(self, embed_mat, row_count, max_length=30):
        """
        embed_mat - embedding matrix for words in training set
        row_count - number of rows in training and validation set
        max_length - maximum length of each tokenized tweet
        """
        self.max_length = max_length
        self.vocab_size = embed_mat.shape[0]
        self.embed_mat = embed_mat
        self.row_count = row_count

    # function to generate chunks of data from csv

    def genData(self, counter=[0], batch_size=32, parent_dir=proc_data_dir):
        """
        Generator function passed to keras.fit_generator to train in chunks
        batch_size - size of data yielded from csv file
        rows - number of rows in file we are generating data from
        counter - keep track of how far into the file we are
        """
        self.batch_size = batch_size
        with open(os.path.join(parent_dir, 'shuffled_processed_data.csv'),
                  'r') as r:
            reader = csv.reader(r)
            # skip header
            next(reader)
            while True:
                x = np.zeros((batch_size, self.max_length))
                x_aux = np.zeros((batch_size, 6))
                y = np.zeros(batch_size)
                for i in range(self.batch_size):
                    try:
                        row = next(reader)
                        # use eval since list will be inside of string
                        x[i] = eval(row[0])
                        x_aux[i] = row[1:7]
                        y[i] = row[7]
                    # reset generator to start of file if we reach end
                    except StopIteration:
                        counter[0] = -self.batch_size
                        break
                counter[0] += self.batch_size
                yield ({'main_input': x, 'aux_input': x_aux},
                       {'main_output': y, 'aux_output': y})

    def fit(self, epochs=5, batch_size=32, lstm_dim=32):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lstm_dim = 32
        # functional API keras implementation of neural network
        main_input = Input(shape=(self.max_length,), dtype='int32',
                           name='main_input')
        embed_layer = Embedding(self.vocab_size, embedding_dim,
                                embeddings_initializer=Constant
                                (self.embed_mat),
                                input_length=self.max_length, trainable=False)(main_input)
        lstm_layer = LSTM(units=self.lstm_dim)(embed_layer)
        # auxilliary output
        auxilliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_layer)
        # auxilliary input
        input_shape = (6,)
        aux_input = Input(input_shape, name='aux_input')
        # concatenate auxilliary input and lstm output
        x = concatenate([lstm_layer, aux_input])
        # pass this through deep neural network
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        main_output = Dense(1, activation='sigmoid', name='main_output')(x)
        model = Model(inputs=[main_input, aux_input], outputs=[main_output,
                      auxilliary_output])
        model.compile(loss='binary_crossentropy', optimizer='rmsprop',
                      metrics=['accuracy'], loss_weights=[0.8, 0.2])
        print(model.summary())
        training_gen = self.genData(batch_size=self.batch_size)
        steps_per_epoch = math.ceil(self.row_count/self.batch_size)
        self.history = model.fit_generator(training_gen,
                                           epochs=self.epochs,
                                           steps_per_epoch=steps_per_epoch)
    # plot graph of validation/training accuracy and loss against epochs

    def plot_graphs(self, history, string):
        plt.plot(history.history[string])
        plt.plot(history.history['val_'+string])
        plt.xlabel("Epochs")
        plt.ylabel(string)
        plt.legend([string, 'val_'+string])
        plt.show()


if __name__ == '__main__':
    # load glove embedding into a dictionary
    embedding_dim = 50
    embed_index = {}
    with open(os.path.join(glove_dir, 'glove.twitter.27B.50d.txt'),
              encoding="UTF-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embed_index[word] = coefs
    # create embedding matrix
    with open(os.path.join(tokenizer_dir, 'tokenizer.pickle'), 'rb') as handle:
        tokenizer = pickle.load(handle)
        word_index = tokenizer.word_index
#        vocab_size = len(word_index) + 1
        vocab_size = 30000
        embed_mat = np.zeros((vocab_size, embedding_dim))
        for word, index in word_index.items():
            embed_vec = embed_index.get(word)
            if embed_vec is not None:
                embed_mat[index-1] = embed_vec
    # numbers/proportions of tweets in original data
    tweet_nos = [1610176, 428542, 1418626, 8377522]
    # get row_count of data
    with open(os.path.join(proc_data_dir, 'shuffled_processed_data.csv'),
              'r') as csvfile:
        csvreader = csv.reader(csvfile)
        row_count = sum(1 for row in csvreader) - 1
    # fit the model
    model = myModel(embed_mat, row_count)
    model.fit()
    model.plot_graphs(model.history, 'main_output_acc')
    model.plot_graphs(model.history, 'main_output_loss')
