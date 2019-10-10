# -*- coding: utf-8 -*-
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input, concatenate
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.initializers import Constant
import pickle
import ast

# number of rows to test with (delete when working with full data,
# and remove df.head(n) from the below segments of code)
# n = 5000

# output dimension of lstm layer
lstm_dim = 32

"""
auxilliary inputs are:
    retweet count
    reply count
    favourite count
    number of hashtags
    number of urls
    number of mentions
"""


# load glove embedding into a dictionary
def load_glove(glove_dir, embedding_dim=50):
    embed_index = {}
    with open(os.path.join(glove_dir, 'glove.twitter.27B.50d.txt'),
              encoding="UTF-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embed_index[word] = coefs
        return(embed_index)


# create embedding matrix
def embedding_matrix(tokenizer_dir, embed_index, vocab_size):
    embedding_dim = list(embed_index.values())[0].shape[0]
    with open(os.path.join(tokenizer_dir, 'tokenizer.pickle'),
              'rb') as handle:
        tokenizer = pickle.load(handle)
        word_index = tokenizer.word_index
        embed_mat = np.zeros((vocab_size, embedding_dim))
        counter = 0
        for word, index in word_index.items():
            embed_vec = embed_index.get(word)
            if embed_vec is not None:
                embed_mat[index-1] = embed_vec
            counter += 1
            if counter == vocab_size:
                break
    return(embed_mat)


def load_data(proc_data_dir, nrows=None):
    """
    proc_data_dir - directory of processed data
    nrows - number of rows of data to load
    """
    with open(os.path.join(proc_data_dir, 'shuffled_processed_data.csv'),
              'r') as r:
        data = pd.read_csv(r, nrows=nrows)
    return(data)


def split_data(data):
    # split tweets into training/validation
    train, test = train_test_split(data,
                                   test_size=0.2,
                                   random_state=4)
    main_Itrain = np.array(train['padded_tweet'].apply(ast.literal_eval).
                           values.tolist())
    aux_Itrain = train.iloc[:, 1:7]
    main_Itest = np.array(test['padded_tweet'].apply(ast.literal_eval).values.
                          tolist())
    aux_Itest = test.iloc[:, 1:7]
    return(main_Itrain, main_Itest, aux_Itrain, aux_Itest, train['label'],
           test['label'])


# functional API keras implementation of neural network
def fit_model(embed_mat, data, vocab_size, max_length, num_epochs, batch_size):
    embedding_dim = embed_mat.shape[1]
    # assign data
    data = split_data(data)
    main_Itrain, main_Itest, aux_Itrain, aux_Itest, train_label, \
        test_label = data
    main_input = Input(shape=(max_length,), dtype='int32', name='main_input')
    embed_layer = Embedding(vocab_size, embedding_dim,
                            embeddings_initializer=Constant(embed_mat),
                            input_length=max_length, trainable=False)(main_input)
    lstm_layer = LSTM(units=lstm_dim)(embed_layer)
    # auxilliary output
    auxilliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_layer)
    # auxilliary input
    input_shape = 6
    aux_input = Input((input_shape,), name='aux_input')
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
    model.summary()
    # all data inputted here must be arrays, not dataframes
    history = model.fit({'main_input': main_Itrain, 'aux_input': aux_Itrain},
                        {'main_output': train_label, 'aux_output':
                            train_label},
                        validation_data=({'main_input': main_Itest,
                                          'aux_input': aux_Itest},
                                         {'main_output': test_label,
                                          'aux_output': test_label}),
                        epochs=num_epochs,
                        batch_size=batch_size)
    return(history)


# plot graph of validation/training accuracy and loss against epochs
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


def run_model(data_dirs, nrows=None, embedding_dim=50, max_length=30,
              num_epochs=10, batch_size=32, vocab_size=30000):
    original_data_dir, tokenizer_dir, proc_data_dir, glove_dir = data_dirs
    # load glove embedding
    embed_index = load_glove(glove_dir)
    # create embedding matrix
    embed_mat = embedding_matrix(tokenizer_dir, embed_index, vocab_size)
    # load data
    data = load_data(proc_data_dir, nrows=nrows)
    # fit model
    history = fit_model(embed_mat, data, vocab_size, max_length, num_epochs,
                        batch_size)
    plot_graphs(history, 'main_output_accuracy')
    plot_graphs(history, 'main_output_loss')
    return(history)
