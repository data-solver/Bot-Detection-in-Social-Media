# -*- coding: utf-8 -*-

import pandas as pd
import lstm_data_processing as ldp
import os
import numpy as np
import matplotlib.pyplot as plt
import math
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
        
vocab_size = len(embed_index)
count = 0
##create embedding matrix
#embed_mat = np.zeros((vocab_size, embedding_dim))
#for word, index in  embed_index.items():
#    embed_vec = embed_index[word]
#    if embed_vec is not None:
#        try:
#            embed_mat[count] = embed_vec  
#            count +=1
#        except ValueError:
#            continue
        
#create embedding matrix based on words in training data
temp_dir = ("C:/Users/Kumar/OneDrive - Imperial College London/Year 3/UROP/"
                  "Dataset/cresci-2017.csv/datasets_full.csv/")
with open(os.path.join(temp_dir, 'social_spambots_1.csv/tweets.csv'), 
          encoding = "Latin-1") as f:
    df = pd.read_csv(f)
    df = df.dropna(subset = ['text'])
tokenized_tweets = []
count = 0
for row in df['text']:
    temp = ldp.tokenizer1(row)
    tokenized_tweets.append(ldp.refine_token(temp))
    count += 1
    if count == 30000:
        break
    
tokenizer = Tokenizer(num_words = 100000, filters = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(tokenized_tweets)
word_index = tokenizer.word_index
#transform our tweets into their integer representation
sequences = tokenizer.texts_to_sequences(tokenized_tweets)

#pad sequences so that they are all same length
#            self.max_length = len(max(sequences))
max_length = 50
tokenised_tweets = pad_sequences(sequences,
                              maxlen = max_length,
                              dtype = 'int32',
                              padding = 'post'
                              )
vocab_size = len(word_index) + 1
embed_mat = np.zeros(shape = (vocab_size, embedding_dim))
for word, index in word_index.items():
    embed_vec = embed_index.get(word)
    if embed_vec is not None:
        embed_mat[index-1] = embed_vec 



class myModel:
    
    def __init__(self, embed_mat, max_length = 50):
        self.max_length = max_length
        self.embed_mat = embed_mat
        self.vocab_size = embed_mat.shape[0]
        tweet_nos = [1610176, 428542, 1418626, 8377522]
        self.total = sum(tweet_nos)
        
        pass
    #function to generate chunks of data from csv
    def genData(self, chunk_size, train_data = True, valid_data = False, counter = [0],
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
        self.chunk_size = chunk_size
        
        proportions = [0,0,0,0]
        
        for index, num in enumerate(tweet_nos):
            proportions[index] = int((num / self.total) * chunk_size)
        
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
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(tokenized_tweets, 
                                                                labels,
                                                                test_size = 0.3,
                                                                random_state = 4)
            
            self.x_aux_train, self.x_aux_test, self.y_aux_train, self.y_aux_test = \
                                                train_test_split(auxilliary_input,
                                                                 labels,
                                                                 test_size = 0.3,
                                                                 random_state = 4)
            #prepare tokenizer
            tokenizer = Tokenizer(num_words = 100000, filters = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
            tokenizer.fit_on_texts(self.x_train)
            self.word_index = tokenizer.word_index
            
            #vocab size + 1 for the out of vocab token
#            self.vocab_size = len(self.word_index) + 1
            
            #transform our tweets into their integer representation
            sequences = tokenizer.texts_to_sequences(self.x_train)
            
            #pad sequences so that they are all same length
#            self.max_length = len(max(sequences))
            x_train = pad_sequences(sequences,
                                          maxlen = self.max_length,
                                          dtype = 'int32',
                                          padding = 'post'
                                          )
            
            #do the same for validation data
    
            testing_sequences = tokenizer.texts_to_sequences(self.x_test)
            self.x_test = pad_sequences(testing_sequences,
                                           maxlen=self.max_length,
                                           dtype = 'int32',
                                           padding = 'post'
                                           )
            
            #create embedding matrix for words in our training data
#            self.embed_mat = np.zeros((self.vocab_size, embedding_dim))
#            for word, index in  self.word_index.items():
#                embed_vec = embed_index.get(word)
#                if embed_vec is not None:
#                    self.embed_mat[index-1] = embed_vec  
            counter[0] += 1
            yield ({'main_input': x_train, 'aux_input': np.asarray(self.x_aux_train)},
            {'main_output': self.y_train, 'aux_output': self.y_aux_train})
    
    def fit(self, chunk_size, epochs = 5, batch_size = 32):
        self.epochs = epochs
        self.batch_size = batch_size
        
        #functional API keras implementation of neural network
        main_input = Input(shape = (self.max_length,), dtype = 'int32', name = 'main_input')
        embed_layer = Embedding(self.vocab_size, embedding_dim, weights = [self.embed_mat],
                                input_length = self.max_length, trainable = False)(main_input)
        lstm_layer = LSTM(units = lstm_dim)(embed_layer)
        
        #auxilliary output
        auxilliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_layer)
        
        #auxilliary input
        input_shape = (6,)
        aux_input = Input( input_shape, name = 'aux_input')
        
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
        
        training_gen = self.genData(chunk_size = chunk_size)
        
        steps_per_epoch = math.ceil(self.total/self.batch_size)
        self.history = model.fit_generator(training_gen, 
                                           epochs = self.epochs,
                                           steps_per_epoch = steps_per_epoch)
#        #all data inputted here must be arrays, not dataframes
#        history = model.fit({'main_input': self.x_train, 'aux_input': np.asarray(self.x_aux_train)},
#                  {'main_output': self.y_train, 'aux_output': self.y_aux_train},
#                  validation_data = ({'main_input': self.x_test, 
#                                      'aux_input': np.asarray(self.x_aux_test)},
#                                      {'main_output': self.y_test,
#                                      'aux_output': self.y_aux_test}),
#                  epochs = num_epochs,
#                  batch_size = batch_size)

    #plot graph of validation/training accuracy and loss against epochs
    def plot_graphs(self, history, string):
      plt.plot(history.history[string])
      plt.plot(history.history['val_'+string])
      plt.xlabel("Epochs")
      plt.ylabel(string)
      plt.legend([string, 'val_'+string])
      plt.show()
        
#plot_graphs(history, 'main_output_acc')
#plot_graphs(history, 'main_output_loss')
