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




#number of rows to test with (delete when working with full data,
#and remove df.head(n) from the below segments of code)
n = 10000
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

"""
problem:
    generate data in batches from csv file
    we have unbalanced classes, so need chuck_size to contain the correct proportion
    make a function which generates chunks of rows from each file
    append the result from each file to an array which contains the current chunk
    drop NA rows - may need to then yield more rows to compensate 
    
    define another function which yields the padded tweets and labels for training
    
    define another function which yields the padded tweets and labels for validation
    
    
    **to do : calculate the correct proportion for chunk_size
    8,377,522 genuine tweets
    3,457,344 bot tweets
    
    let's tentatively set chunk_size = 100,000
    
    defining function
    

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
#function to generate chunks of data from csv
def genData(chunk_size, train_data = True, valid_data = False, counter = [0],
            train_valid_ratio = 0.8):
    """
    Generator function passed to keras.fit_generator to train in chunks
    chunk_size - size of data yielded from csv file
    train_data - if True, yield training data
    valid_data - if True, yield validation data
    train_valid_ratio - ratio of training data to validation data
    """
    
    #proportions of tweets from spambot 1/spambot 2/spambot 3/genuine
    tweet_nos = [1610176, 428542, 1418626, 8377522]
    total = sum(tweet_nos)
    
    proportions = [0,0,0,0]
    
    for index, num in enumerate(tweet_nos):
        proportions[index] = int((num / total) * chunk_size)
    
    #correct errors due to rounding
    total = sum(proportions)
    if total > chunk_size:
        proportions[0] -= total - chunk_size
    if total < chunk_size:
        proportions[0] += chunk_size - total
    while True:
        #social spambot 1
        temp_dir = ("C:/Users/Kumar/OneDrive - Imperial College London/Year 3/UROP/"
              "Dataset/cresci-2017.csv/datasets_full.csv/")
        df = pd.read_csv(os.path.join(temp_dir, 'social_spambots_1.csv/tweets.csv'), 
                          encoding = 'Latin-1', nrows = proportions[0],
                          skiprows = range(1,proportions[0] * counter[0]))
        #social spambot 2
        df = df.append(pd.read_csv(os.path.join(temp_dir, 'social_spambots_2.csv/tweets.csv'), 
                          encoding = 'Latin-1', nrows = proportions[1],
                          skiprows = range(1,proportions[1] * counter[0])))
                                                                        
                        
        #social spambot 3
        df = df.append(pd.read_csv(os.path.join(temp_dir, 'social_spambots_3.csv/tweets.csv'), 
                          encoding = 'Latin-1', nrows = proportions[2],
                          skiprows = range(1,proportions[2] * counter[0])))
        df = df.dropna(subset = ['text'])
        bots_in_chunk = len(df)
        #genuine tweets
        df = df.append(pd.read_csv(os.path.join(temp_dir, 'genuine_accounts.csv/tweets.csv'),
                          encoding = 'Latin-1', nrows = proportions[3],
                          skiprows = range(1,proportions[3] * counter[0])))
        
        #drop any rows with have a NaN entry in text column
        df = df.dropna(subset = ['text'])

        #auxilliary input
        auxilliary_input = df[['reply_count', 'retweet_count',
                              'favorite_count', 'num_hashtags',
                              'num_urls', 'num_mentions']].copy()
        #tokenize the tweets
        tokenized_tweets = []
        for row in df['text']:
            temp = ldp.tokenizer1(row)
            tokenized_tweets.append(ldp.refine_token(temp))
        #labels
        labels = np.zeros(len(df))
        labels[:bots_in_chunk] = 1
        
        #split tweets into training/validation
        x_train, x_test, y_train, y_test = train_test_split(tokenized_tweets, 
                                                            labels,
                                                            test_size = 0.3,
                                                            random_state = 4)
        
        x_aux_train, x_aux_test, y_aux_train, y_aux_test = \
                                            train_test_split(auxilliary_input,
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
        x_train = pad_sequences(sequences,
                                      maxlen = max_length,
                                      dtype = 'int32',
                                      padding = 'post'
                                      )
        
        #create embedding matrix for words in our training data
        embed_mat = np.zeros((vocab_size, embedding_dim))
        for word, index in  word_index.items():
            embed_vec = embed_index.get(word)
            if embed_vec is not None:
                embed_mat[index-1] = embed_vec  
        counter[0] += 1
        yield ({'main_input': x_train, 'aux_input': np.asarray(x_aux_train)},
        {'main_output': y_train, 'aux_output': y_aux_train})
#store our tweets in a list
tokenized_tweets = []


num_dropped = [0,0,0,0]
#bot tweets
#social spambot 1
temp_dir = ("C:/Users/Kumar/OneDrive - Imperial College London/Year 3/UROP/"
          "Dataset/cresci-2017.csv/datasets_full.csv/social_spambots_1.csv")
with open(os.path.join(temp_dir, 'tweets.csv'),encoding = 'Latin-1') as f:
    df = pd.read_csv(f)
    df = df.head(n)
    
    #drop any rows which have a NaN entry in the text column
    df = df.dropna(subset = ['text'])
    num_dropped[0] = n - len(df)
    
    auxilliary_input = df[['reply_count', 'retweet_count',
                              'favorite_count', 'num_hashtags',
                              'num_urls', 'num_mentions']].copy()

for row in df['text']:
    temp = ldp.tokenizer1(row)
    tokenized_tweets.append(ldp.refine_token(temp))

#social spambot 2
temp_dir = ("C:/Users/Kumar/OneDrive - Imperial College London/Year 3/UROP/"
          "Dataset/cresci-2017.csv/datasets_full.csv/social_spambots_2.csv")
with open(os.path.join(temp_dir, 'tweets.csv'),encoding = 'Latin-1') as f:
    df = pd.read_csv(f)
    df = df.head(n)
    df = df.dropna(subset = ['text'])
    num_dropped[1] = n - len(df)

    auxilliary_input = auxilliary_input.append  \
                                (df[['reply_count', 'retweet_count',
                                'favorite_count', 'num_hashtags',
                                'num_urls', 'num_mentions']])

for row in df['text']:
    temp = ldp.tokenizer1(row)
    tokenized_tweets.append(ldp.refine_token(temp))

#social spambot 3
temp_dir = ("C:/Users/Kumar/OneDrive - Imperial College London/Year 3/UROP/"
          "Dataset/cresci-2017.csv/datasets_full.csv/social_spambots_3.csv")
with open(os.path.join(temp_dir, 'tweets.csv'),encoding = 'Latin-1') as f:
    df = pd.read_csv(f)
    df = df.head(n)
    df = df.dropna(subset = ['text'])   
    num_dropped[2] = n - len(df)
    
    auxilliary_input = auxilliary_input.append  \
                                (df[['reply_count', 'retweet_count',
                                'favorite_count', 'num_hashtags',
                                'num_urls', 'num_mentions']])  

for row in df['text']:
    temp = ldp.tokenizer1(row)
    tokenized_tweets.append(ldp.refine_token(temp))
    
#human tweets
temp_dir = ("C:/Users/Kumar/OneDrive - Imperial College London/Year 3/UROP/"
          "Dataset/cresci-2017.csv/datasets_full.csv/genuine_accounts.csv")
with open(os.path.join(temp_dir, 'tweets.csv'),encoding = 'Latin-1') as f:
    df = pd.read_csv(f)
    df = df.head(3*n)
    df = df.dropna(subset = ['text'])
    num_dropped[3] = 3*n - len(df)
    
    auxilliary_input = auxilliary_input.append  \
                                (df[['reply_count', 'retweet_count',
                                'favorite_count', 'num_hashtags',
                                'num_urls', 'num_mentions']])   

for row in df['text']:
    temp = ldp.tokenizer1(row)
    tokenized_tweets.append(ldp.refine_token(temp))
    
#create labels - we have set it up so that the first half of our list of 
#tokenized tweets are bots, and the second half is human - when working with 
#full dataset, this will not be the case
labels = np.zeros(6*n - sum(num_dropped))
labels[:3*n- sum(num_dropped[:-1])] = 1


#split tweets into training/validation
x_train, x_test, y_train, y_test = train_test_split(tokenized_tweets, 
                                                    labels,
                                                    test_size = 0.3,
                                                    random_state = 4)

x_aux_train, x_aux_test, y_aux_train, y_aux_test = \
                                    train_test_split(auxilliary_input,
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
x_train = pad_sequences(sequences,
                              maxlen = max_length,
                              dtype = 'int32',
                              padding = 'post'
                              )

#do the same for validation data

testing_sequences = tokenizer.texts_to_sequences(x_test)
x_test = pad_sequences(testing_sequences,
                               maxlen=max_length,
                               dtype = 'int32',
                               padding = 'post'
                               )

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
        
#create embedding matrix for words in our training data
embed_mat = np.zeros((vocab_size, embedding_dim))
for word, index in  word_index.items():
    embed_vec = embed_index.get(word)
    if embed_vec is not None:
        embed_mat[index-1] = embed_vec  



#functional API keras implementation of neural network
main_input = Input(shape = (max_length,), dtype = 'int32', name = 'main_input')
embed_layer = Embedding(vocab_size, embedding_dim, weights = [embed_mat],
                        input_length = max_length, trainable = False)(main_input)
lstm_layer = LSTM(units = lstm_dim)(embed_layer)

#auxilliary output
auxilliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_layer)

#auxilliary input
input_shape = x_aux_train.shape[1]
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
history = model.fit({'main_input': x_train, 'aux_input': np.asarray(x_aux_train)},
          {'main_output': y_train, 'aux_output': y_aux_train},
          validation_data = ({'main_input': x_test, 
                              'aux_input': np.asarray(x_aux_test)},
                              {'main_output': y_test,
                              'aux_output': y_aux_test}),
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































"""
old sequential implementation
=======



>>>>>>> 62dc9e0572b8bee63ad442f5d2c772bfcaab2ced

history = model.fit(x_train, y_train, epochs=num_epochs, 
                    validation_data=(x_test, y_test))
#implement deep neural network with glove embedding and lstm layer
model = Sequential()   
embed_layer = Embedding(vocab_size, embedding_dim, weights = [embed_mat],
                        input_length = max_length, trainable = False)
model.add(embed_layer)
model.add(LSTM(units = lstm_dim))

#extract output from lstm layer
get_lstm_layer_output = K.function([model.layers[0].input],
                                  [model.layers[1].output])
layer_output = get_lstm_layer_output([x_train])[0]


#concatenate lstm output with auxilliary input

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

num_epochs = 20
history = model.fit(x_train, y_train, epochs=num_epochs, validation_data=(x_test, y_test))
"""
