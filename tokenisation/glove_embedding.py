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

#number of rows to test with
n = 100

#output dimension of lstm layer
lstm_dim = 32


#load glove embedding into a dictionary
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
    
#prepare tokenizer
tokenizer = Tokenizer(num_words = 100000, filters = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(tokenized_tweets)
word_index = tokenizer.word_index

#vocab size + 1 for the out of vocab token
vocab_size = len(word_index) + 1

#transform our tweets into their integer representation
sequences = tokenizer.texts_to_sequences(tokenized_tweets)

#pad sequences so that they are all same length
max_length = len(max(sequences))
padded_tweets = pad_sequences(sequences,
                              maxlen = max_length,
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
