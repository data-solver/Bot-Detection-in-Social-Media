# -*- coding: utf-8 -*-
"""
Contextualised LSTM account level bot detection
"""
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek


class acc_level:
    """
    class for account level bot detection
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        pass

    def load_data(self):
        """
        load and merge account data, add label column for bots/humans
        """
        df1 = pd.DataFrame()
        sub_dirs = ['genuine_accounts.csv', 'social_spambots_1.csv',
                    'social_spambots_2.csv', 'social_spambots_3.csv']
        for entry in sub_dirs:
            path = os.path.join(self.data_dir, entry, 'users.csv')
            with open(path, 'r', encoding="Latin-1") as r:
                users = pd.read_csv(r)
            df1 = df1.append(users, ignore_index=True)
            # count how many human accounts we have in df1
            if 'genuine' in entry:
                human_size = len(df1)
        # append label column for whether or not account is a bot
        bot_size = len(df1) - human_size
        values = human_size * [0] + bot_size * [1]
        y = pd.DataFrame(values, columns=['labels'])
        self.X = df1
        self.y = y

    def fit_randfor(self, test_ratio=0.2, smoteenn=False, smotomek=False):
        """
        fit account level bot detection model
        test ratio - proportion of data split into test set
        smotenn - whether or not to use SMOTENN sampling
        """
        # split data into train and test
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y,
                                                            test_size=0.2,
                                                            random_state=4)
        # if smoteenn is true, use smoteenn sampling
        if smoteenn:
            sme = SMOTEENN(random_state=1)
            X_train, y_train = sme.fit_resample(X_train, y_train)
        # if smotomek is true, use smotomek sampling
        if smotomek:
            smt = SMOTETomek(random_state=1)
            X_train, y_train = smt.fit_resample(X_train, y_train)
            
        # fit the random forest
        clf = RandomForestClassifier(random_state=1)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        return score


if __name__ == '__main__':
    data_dir = r".\Datasets\LSTM paper data\Clean Data"
    model = acc_level(data_dir)
    model.load_data()
    score = model.fit_randfor()
