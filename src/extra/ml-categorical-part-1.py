import os
import copy
import numpy as np
import pandas as pd

from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import FeatureHasher


def main():

    path = r'/Users/jlittler/Documents/Developer/python/mlenv/datasets/kaggle-avazu'
    train = pd.read_csv(os.path.join(path, 'train-10k.csv'))

    msk = np.random.rand(len(train)) < 0.8
    features = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

    # create a simple baseline method
    X_train = train[msk].iloc[:, features]
    X_test = train[~msk].iloc[:, features]
    y_train = train[msk].iloc[:, 1]
    y_test = train[~msk].iloc[:, 1]

    print('{:.2f}'.format(log_loss(y_test, np.ones(len(y_test)) * y_train.mean())))

    # method 1 - encoding to ordinal values
    X_train_ordinal = X_train.values
    X_test_ordinal = X_test.values

    les = []
    l = LogisticRegression()
    r = RandomForestClassifier(n_estimators=25, max_depth=10)

    for i in range(X_train_ordinal.shape[1]):
        le = LabelEncoder()
        le.fit(train.iloc[:, features].iloc[:, i])
        les.append(le)
        X_train_ordinal[:, i] = le.transform(X_train_ordinal[:, i])
        X_test_ordinal[:, i] = le.transform(X_test_ordinal[:, i])

    l.fit(X_train_ordinal, y_train)
    y_pred = l.predict_proba(X_test_ordinal)
    print('{:.2f}'.format(log_loss(y_test, y_pred)))

    r.fit(X_train_ordinal, y_train)
    y_pred = r.predict_proba(X_test_ordinal)
    print('{:.2f}'.format(log_loss(y_test, y_pred)))

    # method 2 - one hot encoding
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(X_train_ordinal)
    X_train_onehot = enc.transform(X_train_ordinal)
    X_test_onehot = enc.transform(X_test_ordinal)

    l.fit(X_train_onehot, y_train)
    y_pred = l.predict_proba(X_test_onehot)
    print('{:.2f}'.format(log_loss(y_test, y_pred)))

    r.fit(X_train_onehot, y_train)
    y_pred = r.predict_proba(X_test_onehot)
    print('{:.2f}'.format(log_loss(y_test, y_pred)))

    # method 3 - group rare values
    X_train_rare = copy.copy(X_train)
    X_test_rare = copy.copy(X_test)
    X_train_rare["test"] = 0
    X_test_rare["test"] = 1

    temp_df = pd.concat([X_train_rare, X_test_rare], axis=0)
    names = list(X_train_rare.columns.values)

    for i in names:
        temp_df.loc[temp_df[i].value_counts()[temp_df[i]].values < 20, i] = 'RARE_VALUE'

    for i in range(temp_df.shape[1]):
        temp_df.iloc[:, i] = temp_df.iloc[:, i].astype('str')

    X_train_rare = temp_df[temp_df['test'] == '0'].iloc[:, :-1].values
    X_test_rare = temp_df[temp_df['test'] == '1'].iloc[:, :-1].values

    for i in range(X_train_rare.shape[1]):
        le = LabelEncoder()
        le.fit(temp_df.iloc[:, :-1].iloc[:, i])
        les.append(le)
        X_train_rare[:, i] = le.transform(X_train_rare[:, i])
        X_test_rare[:, i] = le.transform(X_test_rare[:, i])

    enc.fit(X_train_rare)
    X_train_rare = enc.transform(X_train_rare)
    X_test_rare = enc.transform(X_test_rare)

    l.fit(X_train_rare, y_train)
    y_pred = l.predict_proba(X_test_rare)
    print(log_loss(y_test, y_pred))

    r.fit(X_train_rare, y_train)
    y_pred = r.predict_proba(X_test_rare)
    print(log_loss(y_test, y_pred))

    print(X_train_rare.shape)

    # method 4 - feature hashing
    X_train_hash = copy.copy(X_train)
    X_test_hash = copy.copy(X_test)

    for i in range(X_train_hash.shape[1]):
        X_train_hash.iloc[:, i] = X_train_hash.iloc[:, i].astype('str')

    for i in range(X_test_hash.shape[1]):
        X_test_hash.iloc[:, i] = X_test_hash.iloc[:, i].astype('str')

    h = FeatureHasher(n_features=100, input_type='string')
    X_train_hash = h.transform(X_train_hash.values)
    X_test_hash = h.transform(X_test_hash.values)

    l.fit(X_train_hash, y_train)
    y_pred = l.predict_proba(X_test_hash)
    print(log_loss(y_test, y_pred))

    r.fit(X_train_hash, y_train)
    y_pred = r.predict_proba(X_test_hash)
    print(log_loss(y_test, y_pred))


if __name__ == '__main__':
    main()
