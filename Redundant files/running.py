# -*- coding: utf-8 -*-

import csv
import pandas as pd
import numpy as np
#from sklearn.metrics import confusion_matrix
#results = []
#with open(r'C:\Users\Kumar\OneDrive - Imperial College London\Year 3\UROP\Dataset\cresci-2017.csv\datasets_full.csv\genuine_accounts.csv\users.csv', encoding='utf-8') as csvfile:
#    reader = csv.reader(csvfile, delimiter = ' ',quotechar = '|')
#    for row in reader:
#        results.append(row)
#        
#test = []
#with open(r'C:\Users\Kumar\OneDrive - Imperial College London\Year 3\UROP\Dataset\cresci-2017.csv\datasets_full.csv\genuine_accounts.csv\tweets.csv', encoding = "Latin-1") as f:
#    #remove any null bytes
#    tweets = csv.reader((x.replace('\0', '') for x in f), delimiter = ' ', quotechar = '|')
#    for row in tweets:
#        test.append(row)
        
    
"""
(account level detection)
concatenate all of a user's tweets into a string
do this for every user - getting a list of strings which each represent all of an individual user's tweets
pass this list into obtain_LDA function
run AdaBoost

Repeat, but treat every tweet as a different user (tweet level detection)
"""

with open(r'C:\Users\Kumar\OneDrive - Imperial College London\Year 3\UROP\Dataset\cresci-2017.csv\datasets_full.csv\genuine_accounts.csv\tweets.csv', encoding = "Latin-1") as f:
    df = pd.read_csv(f)
#    df_conc = (df.groupby('user_id', sort = False)['text']
#    .agg(' '.join)
#    .reset_index())
    
#    #change any floats to string (caused an error previously)
#    df.text.apply(str)
    
    #drop any rows which have a NaN entry in the text column
    df = df.dropna(subset = ['text'])
    
#    df.loc['text'] = df.text.astype(str)
    
    #combine the tweets of each user into a string
    df_conc_g = (df.groupby('user_id', sort = False)['text']
    .apply(' '.join)
    .reset_index())

with open(r'C:\Users\Kumar\OneDrive - Imperial College London\Year 3\UROP\Dataset\cresci-2017.csv\datasets_full.csv\social_spambots_1.csv\tweets.csv', encoding = "Latin-1") as f:
    df = pd.read_csv(f)
    df = df.dropna(subset = ['text'])
    df_conc_s1 = (df.groupby('user_id', sort = False)['text']
    .apply(' '.join)
    .reset_index())

with open(r'C:\Users\Kumar\OneDrive - Imperial College London\Year 3\UROP\Dataset\cresci-2017.csv\datasets_full.csv\social_spambots_2.csv\tweets.csv', encoding = "Latin-1") as f:
    df = pd.read_csv(f)
    df = df.dropna(subset = ['text'])
    df_conc_s2 = (df.groupby('user_id', sort = False)['text']
    .apply(' '.join)
    .reset_index())

with open(r'C:\Users\Kumar\OneDrive - Imperial College London\Year 3\UROP\Dataset\cresci-2017.csv\datasets_full.csv\social_spambots_3.csv\tweets.csv', encoding = "Latin-1") as f:
    df = pd.read_csv(f)
    df = df.dropna(subset = ['text'])
    df_conc_s3 = (df.groupby('user_id', sort = False)['text']
    .apply(' '.join)
    .reset_index())

tweet_list = []
for element in df_conc_g['text']:
    tweet_list.append(element)
for element in df_conc_s1['text']:
    tweet_list.append(element)
for element in df_conc_s2['text']:
    tweet_list.append(element)
for element in df_conc_s3['text']:
    tweet_list.append(element)
    
#set max number of features
max_features = 20

lda_transformed, feature_names = obtain_LDA(tweet_list, max_features = max_features)

#create vector y of labels corresponding 1 if bot, -1 if not
gen_no = df_conc_g.shape[0]
bot_no = df_conc_s1.shape[0] + df_conc_s2.shape[0] + df_conc_s3.shape[0]
y1 = np.repeat(-1, gen_no)
y2 = np.repeat(1, bot_no)

y = np.concatenate([y1,y2])

#get predictions from our AdaBoost model
pred_ensemble = AdaBoost(lda_transformed, y)


#create confusion matrix to measure how well our model performed
correct_preds = np.sum(y == pred_ensemble)
y_act = pd.Series(y, name = 'Actual')
y_pred = pd.Series(pred_ensemble, name = 'Prediction')

confusion_mat = pd.crosstab(y_act, y_pred)

t_pos = confusion_mat.iloc[1,1]
f_pos = confusion_mat.iloc[0,1]
t_neg = confusion_mat.iloc[0,0]
f_neg = confusion_mat.iloc[1,0]

#compute different measures of model performance
precision = t_pos / (t_pos + f_neg)
recall = t_neg / (t_neg + f_pos)
F1 = (2 * precision * recall)/(precision + recall)

"""
comments: how to use extra features other than tweet contents in AdaBoost?

"""

"""
now we do tweet - level classification

combine all tweets into a large list
"""
with open(r'C:\Users\Kumar\OneDrive - Imperial College London\Year 3\UROP\Dataset\cresci-2017.csv\datasets_full.csv\genuine_accounts.csv\tweets.csv', encoding = "Latin-1") as f:
    df = pd.read_csv(f)
    df = df.dropna(subset = ['text'])
    
    #keep track of number of bot/genuine tweets
    gen_no = df.shape[0]
    bot_no = 0
    
tweet_list_t = []
tweet_list_t = df['text']

with open(r'C:\Users\Kumar\OneDrive - Imperial College London\Year 3\UROP\Dataset\cresci-2017.csv\datasets_full.csv\social_spambots_1.csv\tweets.csv', encoding = "Latin-1") as f:
    df = pd.read_csv(f)
    df = df.dropna(subset = ['text'])
    bot_no += df.shape[0]
tweet_list_t.append(df['text'])
    
with open(r'C:\Users\Kumar\OneDrive - Imperial College London\Year 3\UROP\Dataset\cresci-2017.csv\datasets_full.csv\social_spambots_2.csv\tweets.csv', encoding = "Latin-1") as f:
    df = pd.read_csv(f)
    df = df.dropna(subset = ['text'])
    bot_no += df.shape[0]
tweet_list_t.append(df['text'])

with open(r'C:\Users\Kumar\OneDrive - Imperial College London\Year 3\UROP\Dataset\cresci-2017.csv\datasets_full.csv\social_spambots_3.csv\tweets.csv', encoding = "Latin-1") as f:
    df = pd.read_csv(f)
    df = df.dropna(subset = ['text'])
    bot_no += df.shape[0]
tweet_list_t.append(df['text'])

"""
Obtain LDA using this large tweet list - problem - memory error 
"""

lda_transformed_t, feature_names_t = obtain_LDA(tweet_list_t, max_features)

#create vector of labels corresponding to 1 if bot, -1 if human

y1 = np.repeat(-1, gen_no)
y2 = np.repeat(1, bot_no)
y_t = np.concatenate([y1,y2])

pred_ensemble_t = AdaBoost(lda_transformed_t, y_t)

#create confusion matrix to measure how well our model performed
correct_preds_t = np.sum(y == pred_ensemble_t)
y_act_t = pd.Series(y_t, name = 'Actual')
y_pred_t = pd.Series(pred_ensemble_t, name = 'Prediction')

confusion_mat_t = pd.crosstab(y_act_t, y_pred_t)

t_pos = confusion_mat_t.iloc[1,1]
f_pos = confusion_mat_t.iloc[0,1]
t_neg = confusion_mat_t.iloc[0,0]
f_neg = confusion_mat_t.iloc[1,0]

#compute different measures of model performance
precision = t_pos / (t_pos + f_neg)
recall = t_neg / (t_neg + f_pos)
F1 = (2 * precision * recall)/(precision + recall)


