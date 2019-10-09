# -*- coding: utf-8 -*-
from AdaBoost import AdaBoost
from data_preprocessing import lstm_data_processing as ldp, preprocessed_data
from LSTM import contextualised_lstm, contextualised_lstm_gen

if __name__ == "__main__":
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
    data_dirs = [original_data_dir, tokenizer_dir, proc_data_dir]
    preprocessed_data.run_processing(num_words=30000, length=100000,
                                     data_dirs=data_dirs, new_tokenizer=False,
                                     proc_data=False, shuffle=False)
    # AdaBoost without SMOTENN sampling
    AdaBoost_model = AdaBoost.AdaBoost(proc_data_dir, SMOTENN=False)
    clf, score = AdaBoost_model.fit()
    print(score, "without SMOTENN sampling")
    # AdaBoost with SMOTENN sampling
    AdaBoost_model_SMOTENN = AdaBoost.AdaBoost(proc_data_dir, SMOTENN=True)
    clf, score = AdaBoost_model_SMOTENN.fit()
    print(score, "with SMOTENN sampling")
