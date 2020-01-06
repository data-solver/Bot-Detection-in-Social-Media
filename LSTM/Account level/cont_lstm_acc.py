# -*- coding: utf-8 -*-
"""
Contextualised LSTM account level bot detection
"""
import pandas as pd
import os

class cont_lstm_model:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
    def load_data(self):
        df1 = pd.DataFrame()
        sub_dirs = ['genuine_accounts.csv', 'social_spambots_1.csv', 
                    'social_spambots_2.csv', 'social_spambots_3.csv']
        for entry in sub_dirs:
            path = os.path.join(self.data_dir, entry, 'users.csv')
            with open(path, 'r', encoding="Latin-1") as r:
                users = pd.read_csv(r)
            df1 = df1.append(users)
        self.df1 = df1
        return None
    def fit(self):
        pass
    
if __name__ == '__main__':
    data_dir = r".\Datasets\LSTM paper data\Clean Data"
    model = cont_lstm_model(data_dir)
    model.load_data()