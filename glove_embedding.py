# -*- coding: utf-8 -*-

"""
use 50D pre-trained vector

assume the variable tweet is the tweet we are passing through the embedding
"""
import os
import numpy as np

tweet = "Testing one two three."

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

embedding_dim = 50
counter = 0
vocab_length = 50000
embeddings_index = {}
GLOVE_DIR = r"C:\Users\Kumar\OneDrive - Imperial College London\Year 3\UROP\glove.twitter.27B"
with open(os.path.join(GLOVE_DIR, 'glove.twitter.27B.50d.txt'), encoding = "UTF-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
        
        #limit vocabulary to vocabulary_length
        counter += 1
        if counter == vocab_length:
            break



