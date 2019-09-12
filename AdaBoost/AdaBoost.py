# -*- coding: utf-8 -*-
"""

@author: Kumar
"""

import numpy as np
import os,csv,json,time
from os import listdir
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.tree import DecisionTreeClassifier

def obtain_LDA(list_tweets,max_features):
    # CountVectorizer implements both tokenization and occurence counting
    # currently implementing stop-words for english only
    # default config tokenizes by extracting words of at least 2 letters
    
    """
    need stop words in multiple languages
    """
    vectorizer = CountVectorizer(input='string',stop_words='english',max_features=max_features,min_df=2)
    document_term = vectorizer.fit_transform(list_tweets)
    feature_names = vectorizer.get_feature_names()
    # lda with number of topics = 200
    lda = LatentDirichletAllocation(n_components=200)
    lda_transformed = lda.fit_transform(document_term)
    return lda_transformed,feature_names

def AdaBoost(X,y):
    """
    receive input from Latent Dirichlet Allocation
    X would be the matrix with rows as users and columns as the
    distribution of their tweets over the 200 topics identified by LDA
    y would be the labels i.e. bot or not which would correspond to 1 and -1
    """
    max_iter = 100
    N = X.shape[0]
    d = np.ones(N)/N ## assume initial instances weights are all equal i.e. 1
    y_final = np.zeros(N)
    ## intial iteration
    h = DecisionTreeClassifier(max_depth=1,splitter='random')
    h.fit(X,y) # samples equally weighted initially
    pred = h.predict(X)
    epsilon = d.dot(pred!=y)
    beta = epsilon/(1-epsilon)
    alpha = -(np.log(beta))/2
    d = d*np.exp(-alpha*y*pred)/d.sum() # instance weights for second iteration
    y_final += alpha*pred
    iteration = 1
    conv_criteria = np.dot(pred!=y,np.ones(len(y)))/len(y)
    ## subsequent iterations
    while iteration <= max_iter and conv_criteria >= 1e-2:
        ## train the classifier
        h = DecisionTreeClassifier(max_depth=1,splitter='random')
        h.fit(X,y,sample_weight=d)
        ## get predictions on all data points and convergence criteria
        prev_pred = pred
        pred = h.predict(X)
        conv_criteria = np.dot(pred!=prev_pred,np.ones(len(y)))/len(y)
        ## get classifier weight
        epsilon = d.dot(pred!=y)/d.sum()
        beta = epsilon/(1-epsilon)
        alpha = -(np.log(beta))/2
        ## add weighted classifer to previous classifiers
        y_final += alpha*pred
        ## calculate instance weights for next iteration
        d = d*np.exp(-alpha*y*pred)/d.sum()
        iteration+=1
    pred_ensemble = np.sign(y_final)
    return pred_ensemble