# -*- coding: utf-8 -*-

import pandas as pd
import lstm_data_processing as ldp
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Flatten, Dense, Embedding, LSTM, Input, \
                                    concatenate
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.initializers import Constant
import pickle
import ast




#number of rows to test with (delete when working with full data,
#and remove df.head(n) from the below segments of code)
n = 5000

#output dimension of lstm layer
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
                                                    
#load glove embedding into a dictionary
embedding_dim = 50
embed_index = {}
GLOVE_DIR = r"C:\Users\Kumar\OneDrive - Imperial College London\Year 3\UROP\glove.twitter.27B"
with open(os.path.join(GLOVE_DIR, 'glove.twitter.27B.50d.txt'), encoding = "UTF-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embed_index[word] = coefs

#create embedding matrix
parent_dir = ("C:/Users/Kumar/OneDrive - Imperial College London/"
              "Github repositories/Bot-Detection-in-Social-Media/tokenisation")
with open(os.path.join(parent_dir,'tokenizer.pickle'), 'rb') as handle:
    tokenizer = pickle.load(handle)
    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1
    embed_mat = np.zeros((vocab_size, embedding_dim))
    for word, index in word_index.items():
        embed_vec = embed_index.get(word)
        if embed_vec is not None:
            embed_mat[index-1] = embed_vec


parent_dir = ("C:/Users/Kumar/OneDrive - Imperial College London/Year 3/UROP"
              "/Dataset")
with open(os.path.join(parent_dir, 'shuffled_processed_data.csv'), 'r') as r:
    data = pd.read_csv(r, nrows = n)
#split tweets into training/validation
train, test = train_test_split(data,
                               test_size = 0.2,
                               random_state = 4)
main_Itrain = np.array(train['padded_tweet'].apply(ast.literal_eval).values.tolist())
aux_Itrain = train.iloc[:,1:7]
main_Itest = np.array(test['padded_tweet'].apply(ast.literal_eval).values.tolist())
aux_Itest = test.iloc[:,1:7]


max_length = 200
#functional API keras implementation of neural network
main_input = Input(shape = (max_length,), dtype = 'int32', name = 'main_input')
embed_layer = Embedding(vocab_size, embedding_dim,
                        embeddings_initializer = Constant(embed_mat),
                        input_length = max_length, trainable = False)(main_input)
lstm_layer = LSTM(units = lstm_dim)(embed_layer)

#auxilliary output
auxilliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_layer)

#auxilliary input
input_shape = 6
aux_input = Input( (input_shape,), name = 'aux_input')

#concatenate auxilliary input and lstm output
x = concatenate([lstm_layer, aux_input])

#pass this through deep neural network
x = Dense(128, activation = 'relu')(x)
x = Dense(64, activation = 'relu')(x)
main_output = Dense(1, activation = 'sigmoid', name = 'main_output')(x)

model = Model(inputs = [main_input, aux_input], outputs = [main_output,
              auxilliary_output])
model.compile(loss='binary_crossentropy',optimizer='rmsprop',
              metrics=['accuracy'], loss_weights = [0.8, 0.2])
model.summary()

num_epochs = 20
batch_size = 32

#all data inputted here must be arrays, not dataframes
history = model.fit({'main_input': main_Itrain, 'aux_input': aux_Itrain},
          {'main_output': train['label'], 'aux_output': train['label']},
          validation_data = ({'main_input': main_Itest, 
                              'aux_input': aux_Itest},
                              {'main_output': test['label'],
                              'aux_output': test['label']}),
          epochs = num_epochs,
          batch_size = batch_size)

#plot graph of validation/training accuracy and loss against epochs
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

plot_graphs(history, 'main_output_acc')
plot_graphs(history, 'main_output_loss')