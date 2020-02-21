# -*- coding: utf-8 -*-
from AdaBoost import AdaBoost
from data_preprocessing import preprocessed_data
from LSTM.Tweet_level import cont_lstm_tweet as cl
from LSTM.Account_level import cont_lstm_acc
from Sentiment_Analysis import sent_acc, sent_tweet


def run_models(process=False, ada=False, SMOTENN=False, lstm=False, sent=False):
    """
    ada - Boolean, whether or not to run regular AdaBoost
    SMOTENN - Boolean, whether or not to run AdaBoost with SMOTENN
              sampling
    lstm - Boolean, whether or not to run lstm model (RAM intensive approach)
    sent - whether or not to run sentiment analysis
    length - Integer, number of rows of data to work with
    """
    # directories for files and data
    # original data
    original_data_dir = ("./Datasets/LSTM paper data/Clean Data")
    # tokenizer fit on training set
    tokenizer_dir = ("./Datasets/LSTM paper data/Tokenizer")
    # pre-processed data
    proc_data_dir = ("./Datasets/LSTM paper data/Processed Data")
    # glove embedding
    glove_dir = ("./Datasets/LSTM paper data/glove.twitter.27B")
    data_dirs = [original_data_dir, tokenizer_dir, proc_data_dir, glove_dir]
    num_words = 50000
    if process:
        preprocessed_data.run_processing(num_words=num_words, length=False,
                                         data_dirs=data_dirs, new_tokenizer=False,
                                         proc_data=False, shuffle=True)
    if ada:
        # AdaBoost without SMOTENN sampling
        AdaBoost_model = AdaBoost.AdaBoost(proc_data_dir, SMOTENN=False)
        clf, score = AdaBoost_model.fit()
        print(score, "without SMOTENN sampling")
    if SMOTENN:
        # AdaBoost with SMOTENN sampling
        AdaBoost_model_SMOTENN = AdaBoost.AdaBoost(proc_data_dir, SMOTENN=True)
        clf, score_SMOTENN = AdaBoost_model_SMOTENN.fit()
        print(score_SMOTENN, "with SMOTENN sampling")
    if lstm:
        # contextualised LSTM (tweet level)
        model = cl.run_model(data_dirs, num_epochs=3, vocab_size=num_words)
        # save model
        model.save('./Datasets/LSTM paper data/Saved models/'
                   'contextualised_lstm_nogen.h5')
        # account level bot detection, with out of box methods
        cont_lstm_acc.run_model()
    if sent:
        # account level sentiment analysis, varying number of tweets considered
        sent_acc.run_model()
        # tweet level sentiment analysis
        sent_tweet.run_model()

if __name__ == '__main__':
    run_models(process=False)