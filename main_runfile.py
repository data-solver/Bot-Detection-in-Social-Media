# -*- coding: utf-8 -*-
from AdaBoost import AdaBoost
from data_preprocessing import preprocessed_data
from LSTM import contextualised_lstm as cl, contextualised_lstm_gen as clg

def run_models(AdaBoost, SMOTENN, lstm, lstm_gen, length):
    """
    AdaBoost - Boolean, whether or not to run regular AdaBoost
    SMOTENN - Boolean, whether or not to run AdaBoost with SMOTENN 
              sampling
    lstm - Boolean, whether or not to run lstm model (RAM intensive approach)
    lstm_gen - Boolean, whether or not to run lstm model (generator approach)
    length - Integer, number of rows of data to work with
    """
    # directories for files and data
    # original data
    original_data_dir = ("C:/Users/Kumar/OneDrive - Imperial College London/"
                         "Year 3/UROP/Dataset/cresci-2017.csv/"
                         "datasets_full.csv/")
    # tokenizer fit on training set
    tokenizer_dir = ("C:/Users/Kumar/OneDrive - Imperial College London/"
                     "Github repositories/Bot-Detection-in-Social-Media/"
                     "Tokenizer")
    # pre-processed data
    proc_data_dir = ("C:/Users/Kumar/OneDrive - Imperial College London/"
                     "Year 3/UROP/Dataset")
    # glove embedding
    glove_dir = ("C:/Users/Kumar/OneDrive - Imperial College London/Year 3/"
                 "UROP/glove.twitter.27B")
    data_dirs = [original_data_dir, tokenizer_dir, proc_data_dir, glove_dir]
    num_words = 50000
    preprocessed_data.run_processing(num_words=num_words, length=False,
                                     data_dirs=data_dirs, new_tokenizer=True,
                                     proc_data=False, shuffle=True)
    # AdaBoost without SMOTENN sampling
    AdaBoost_model = AdaBoost.AdaBoost(proc_data_dir, SMOTENN=False)
    clf, score = AdaBoost_model.fit()
    print(score, "without SMOTENN sampling")
    # AdaBoost with SMOTENN sampling
    AdaBoost_model_SMOTENN = AdaBoost.AdaBoost(proc_data_dir, SMOTENN=True)
    clf, score_SMOTENN = AdaBoost_model_SMOTENN.fit()
    print(score_SMOTENN, "with SMOTENN sampling")

    # contextualised LSTM, without generator for data
    model = cl.run_model(data_dirs, num_epochs=3, vocab_size=num_words)
    # save model
    model.save('./Saved models/contextualised_lstm_nogen.h5')
    # contextualised LSTM, with generator for data
    history = clg.run_model(data_dirs)
    return(history)

if __name__ == '__main__':
    pass