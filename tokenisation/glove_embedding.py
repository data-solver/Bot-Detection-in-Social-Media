# -*- coding: utf-8 -*-

import pandas as pd
import lstm_data_processing as ldp
import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



#number of rows to test with (delete when working with full data,
#and remove df.head(n) from the below segments of code)
n = 1000

#output dimension of lstm layer
lstm_dim = 32


#store our tweets in a list
tokenized_tweets = []

#bot tweets
#social spambot 1
with open(r'C:\Users\Kumar\OneDrive - Imperial College London\Year 3\UROP\\'
          r'Dataset\cresci-2017.csv\datasets_full.csv\social_spambots_1.csv'
          r'\tweets.csv',
          encoding = 'Latin-1') as f:
    df = pd.read_csv(f)
    df = df.head(n)

for row in df['text']:
    temp = ldp.tokenizer1(row)
    tokenized_tweets.append(ldp.refine_token(temp))

#social spambot 2
with open(r'C:\Users\Kumar\OneDrive - Imperial College London\Year 3\UROP\\'
          r'Dataset\cresci-2017.csv\datasets_full.csv\social_spambots_2.csv'
          r'\tweets.csv',
          encoding = 'Latin-1') as f:
    df = pd.read_csv(f)
    df = df.head(n)

for row in df['text']:
    temp = ldp.tokenizer1(row)
    tokenized_tweets.append(ldp.refine_token(temp))

#social spambot 3
with open(r'C:\Users\Kumar\OneDrive - Imperial College London\Year 3\UROP\\'
          r'Dataset\cresci-2017.csv\datasets_full.csv\social_spambots_3.csv'
          r'\tweets.csv',
          encoding = 'Latin-1') as f:
    df = pd.read_csv(f)
    df = df.head(n)

for row in df['text']:
    temp = ldp.tokenizer1(row)
    tokenized_tweets.append(ldp.refine_token(temp))
    
#human tweets
with open(r'C:\Users\Kumar\OneDrive - Imperial College London\Year 3\UROP\\'
          r'Dataset\cresci-2017.csv\datasets_full.csv\genuine_accounts.csv'
          r'\tweets.csv',
          encoding = 'Latin-1') as f:
    df = pd.read_csv(f)
    df = df.head(3*n)

for row in df['text']:
    temp = ldp.tokenizer1(row)
    tokenized_tweets.append(ldp.refine_token(temp))
    
#create labels - we have set it up so that the first half of our list of 
#tokenized tweets are bots, and the second half is human - when working with 
#full dataset, this will not be the case
labels = np.zeros(6*n)
labels[:3*n] = 1

x_train, x_test, y_train, y_test = train_test_split(tokenized_tweets, 
                                                    labels,
                                                    test_size = 0.3,
                                                    random_state = 4)



#prepare tokenizer
tokenizer = Tokenizer(num_words = 100000, filters = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(x_train)
word_index = tokenizer.word_index

#vocab size + 1 for the out of vocab token
vocab_size = len(word_index) + 1

#transform our tweets into their integer representation
sequences = tokenizer.texts_to_sequences(x_train)

#pad sequences so that they are all same length
max_length = len(max(sequences))
padded_tweets = pad_sequences(sequences,
                              maxlen = max_length,
                              dtype = 'int32',
                              padding = 'post'
                              )

#do the same for validation data

testing_sequences = tokenizer.texts_to_sequences(x_test)
testing_padded = pad_sequences(testing_sequences,
                               maxlen=max_length,
                               dtype = 'int32',
                               padding = 'post'
                               )

#load glove embedding into a dictionary
embedding_dim = 50
vocab_length = 50000
embed_index = {}
GLOVE_DIR = r"C:\Users\Kumar\OneDrive - Imperial College London\Year 3\UROP\glove.twitter.27B"
with open(os.path.join(GLOVE_DIR, 'glove.twitter.27B.50d.txt'), encoding = "UTF-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embed_index[word] = coefs
        
#create embedding matrix for words in our training data
embed_mat = np.zeros((vocab_size, embedding_dim))
for word, index in  word_index.items():
    embed_vec = embed_index.get(word)
    if embed_vec is not None:
        embed_mat[index-1] = embed_vec  

#implement deep neural network with glove embedding and lstm layer
model = Sequential()   
embed_layer = Embedding(vocab_size, embedding_dim, weights = [embed_mat],
                        input_length = max_length, trainable = False)
model.add(embed_layer)
model.add(LSTM(units = lstm_dim))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

num_epochs = 20
history = model.fit(padded_tweets, y_train, epochs=num_epochs, validation_data=(testing_padded, y_test))

#plot graph of validation/training accuracy and loss against epochs
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

plot_graphs(history, 'acc')
plot_graphs(history, 'loss')