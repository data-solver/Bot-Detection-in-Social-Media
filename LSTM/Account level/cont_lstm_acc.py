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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import SGDClassifier
from keras.models import Sequential
from keras.layers import Dense


class acc_level:
    """
    class for account level bot detection
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir

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

    def prep_data(self, test_ratio, smoteenn, smotomek):
        # split data into train and test
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y,
                                                            test_size=test_ratio,
                                                            random_state=4)
        # if smoteenn is true, use smoteenn sampling
        if smoteenn:
            sme = SMOTEENN(random_state=1)
            X_train, y_train = sme.fit_resample(X_train, y_train)
        # if smotomek is true, use smotomek sampling
        if smotomek:
            smt = SMOTETomek(random_state=1)
            X_train, y_train = smt.fit_resample(X_train, y_train)
        return X_train, X_test, y_train, y_test

    def fit_randfor(self, test_ratio=0.2, smoteenn=False, smotomek=False):
        """
        fit random forest model
        test ratio - proportion of data split into test set
        smotenn - whether or not to use SMOTENN sampling
        smotomek - whether or not to use SMOTOMEK sampling
        """
        # split data into train and test
        X_train, X_test, y_train, y_test = self.prep_data(test_ratio, smoteenn,
                                                          smotomek)
        # fit the random forest
        clf = RandomForestClassifier(random_state=1)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        return score

    def fit_adaboost(self, test_ratio=0.2, smoteenn=False, smotomek=False):
        """
        fit AdaBoost model
        """
        # split data into train and test
        X_train, X_test, y_train, y_test = self.prep_data(test_ratio, smoteenn,
                                                          smotomek)
        clf = AdaBoostClassifier(random_state=1)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        return score

    def fit_log_reg(self, test_ratio=0.2, smoteenn=False, smotomek=False):
        """
        fit logistic regression
        """
        # split data into train and test
        X_train, X_test, y_train, y_test = self.prep_data(test_ratio, smoteenn,
                                                          smotomek)
        # standardise our data
        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)
        # fit logistic regression
        clf = LogisticRegressionCV(cv=10, random_state=1, penalty='l2',
                                   refit=True).fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        return score

    def fit_sgd(self, test_ratio=0.2, smoteenn=False, smotomek=False):
        """
        fit stochastic gradient descent classifier
        """
        # split data into train and test
        X_train, X_test, y_train, y_test = self.prep_data(test_ratio, smoteenn,
                                                          smotomek)
        # fit sgd classifier
        clf = SGDClassifier(max_iter=1000, tol=1e-3)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        return score

    def fit_adam_nn(self, layer_sizes, test_ratio=0.2, smoteenn=False,
                    smotomek=False):
        """
        fit neural network using adam optimiser, with 2 layers. The size of the
        layers is to be specified as input
        layer_sizes - list of layer sizes with 3 entries
        """
        # split data into train and test
        X_train, X_test, y_train, y_test = self.prep_data(test_ratio, smoteenn,
                                                          smotomek)
        # create neural network
        model = Sequential()
        model.add(Dense(layer_sizes[0], input_dim=X_train.shape[1],
                        activation='relu'))
        model.add(Dense(layer_sizes[1], activation='relu'))
        model.add(Dense(layer_sizes[2], activation='sigmoid'))
        # compile model
        model.compile(loss='binary_crossentropy', optimizer='adam',
                      metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, batch_size=16)
        score, acc = model.evaluate(X_test, y_test, verbose=0)
        return acc


if __name__ == '__main__':
    data_dir = r".\Datasets\LSTM paper data\Clean Data"
    model = acc_level(data_dir)
    model.load_data()
    scores = {}
    scores['Logistic Regression'] = model.fit_log_reg()
    scores['Logistic Regression + SMOTENN'] = model.fit_log_reg(smoteenn=True)
    scores['Logistic Regression + SMOTOMEK'] = model.fit_log_reg(smotomek=True)
    scores['SGD Classifier'] = model.fit_sgd()
    scores['SGD Classifier + SMOTENN'] = model.fit_sgd(smoteenn=True)
    scores['SGD Classifier + SMOTOMEK'] = model.fit_sgd(smotomek=True)
    scores['Random Forest'] = model.fit_randfor()
    scores['Random Forest + SMOTENN'] = model.fit_randfor(smoteenn=True)
    scores['Random Forest + SMOTOMEK'] = model.fit_randfor(smotomek=True)
    scores['AdaBoost'] = model.fit_adaboost()
    scores['AdaBoost + SMOTENN'] = model.fit_adaboost(smoteenn=True)
    scores['AdaBoost + SMOTOMEK'] = model.fit_adaboost(smotomek=True)
    layer_sizes = [500, 200, 1]
    scores['2-layer NN (500,200,1) Relu+Adam'] = model.fit_adam_nn(layer_sizes)
    layer_sizes = [300, 200, 1]
    scores['2-layer NN (500,200,1) Relu+Adam + SMOTENN'] = \
        model.fit_adam_nn(layer_sizes, smoteenn=True)
    scores['2-layer NN (500,200,1) Relu+Adam + SMOTOMEK'] = \
        model.fit_adam_nn(layer_sizes, smotomek=True)
    for key, val in scores.items():
        print(key, val)
