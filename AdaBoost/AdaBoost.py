# -*- coding: utf-8 -*-
import os
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from imblearn.combine import SMOTEENN


class AdaBoost:
    def __init__(self, data_dir, nrows=None, SMOTENN=False):
        """
        data_dir - directory of dataset
        nrows - number of rows of data to work with
        SMOTENN - whether or not to use SMOTENN sampling
        """
        self.data_dir = data_dir
        self.nrows = nrows
        self.SMOTENN = SMOTENN
        pass

    def get_data(self):
        """
        function to fetch data
        """
        with open(os.path.join(self.data_dir, 'shuffled_processed_data.csv'),
                  'r') as r:
            data = pd.read_csv(r, nrows=self.nrows)
            X = data.iloc[:, 1:7]
            y = data.iloc[:, 7]
            if self.SMOTENN:
                sm = SMOTEENN(random_state=0)
                X, y = sm.fit_sample(X, y)
            return(X, y)

    def fit(self):
        """
        fits the AdaBoost model
        """
        clf = AdaBoostClassifier(random_state=0)
        X, y = self.get_data()
        clf.fit(X, y)
        score = clf.score(X, y)
        return(clf, score)

    def graph(self):
        pass


if __name__ == '__main__':
    proc_data_dir = ("C:/Users/Kumar/OneDrive - Imperial College London/"
                     "Year 3/UROP/Dataset")
    model = AdaBoost(proc_data_dir, SMOTENN=True)
    clf, score = model.fit()
    print(score)
