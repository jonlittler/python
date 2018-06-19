import os
import copy
import numpy as np
import pandas as pd

from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from random import shuffle
from gensim.models.word2vec import Word2Vec

# https://blog.myyellowroad.com/using-categorical-data-in-machine-learning-with-python-from-dummy-variables-to-deep-category-42fd0a43b009


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

    l = LogisticRegression()
    r = RandomForestClassifier(n_estimators=25, max_depth=10)

    # method 5 - encoding with dataset stats
    X_train_count = copy.copy(X_train)
    X_test_count = copy.copy(X_test)
    X_train_count['test'] = 0
    X_test_count['test'] = 1

    temp_df = pd.concat([X_train_count, X_test_count], axis=0)

    for i in range(temp_df.shape[1]):
        temp_df.iloc[:, i] = temp_df.iloc[:, i].astype('category')

    X_train_count = temp_df[temp_df['test'] == 0].iloc[:, :-1]
    X_test_count = temp_df[temp_df['test'] == 1].iloc[:, :-1]

    X_train_count.iloc[:, 1].value_counts()

    for i in range(X_train_count.shape[1]):
        counts = X_train_count.iloc[:, i].value_counts()
        counts = counts.sort_index()
        counts = counts.fillna(0)
        counts += np.random.rand(len(counts)) / 1000
        X_train_count.iloc[:, i].cat.categories = counts
        X_test_count.iloc[:, i].cat.categories = counts

    l.fit(X_train_count, y_train)
    y_pred = l.predict_proba(X_test_count)
    print('{:.2f}'.format(log_loss(y_test, y_pred)))

    r.fit(X_train_count, y_train)
    y_pred = r.predict_proba(X_test_count)
    print('{:.2f}'.format(log_loss(y_test, y_pred)))

    # method 6 - encoding with click thru rate
    X_train_ctr = copy.copy(X_train)
    X_test_ctr = copy.copy(X_test)
    X_train_ctr['test'] = 0
    X_test_ctr['test'] = 1

    temp_df = pd.concat([X_train_ctr, X_test_ctr], axis=0)

    for i in range(temp_df.shape[1]):
        temp_df.iloc[:, i] = temp_df.iloc[:, i].astype('category')

    X_train_ctr = temp_df[temp_df['test'] == 0].iloc[:, :-1]
    X_test_ctr = temp_df[temp_df['test'] == 1].iloc[:, :-1]

    temp_df = pd.concat([X_train_ctr, y_train], axis=1)
    names = list(X_train_ctr.columns.values)

    for i in names:
        means = temp_df.groupby(i)['click'].mean()
        means = means.fillna(sum(temp_df['click']) / len(temp_df['click']))
        means += np.random.rand(len(means)) / 1000
        X_train_ctr[i].cat.categories = means
        X_test_ctr[i].cat.categories = means

    l.fit(X_train_ctr, y_train)
    y_pred = l.predict_proba(X_test_ctr)
    print('{:.2f}'.format(log_loss(y_test, y_pred)))

    r.fit(X_train_ctr, y_train)
    y_pred = r.predict_proba(X_test_ctr)
    print('{:.2f}'.format(log_loss(y_test, y_pred)))

    # method 6 - cat2vec
    size = 6
    window = 8

    x_w2v = copy.deepcopy(train.iloc[:, features])
    names = list(x_w2v.columns.values)

    for i in names:
        x_w2v[i] = x_w2v[i].astype('category')
        x_w2v[i].cat.categories = ['Feature %s %s' % (i, g) for g in x_w2v[i].cat.categories]

    x_w2v = x_w2v.values.tolist()
    for i in x_w2v:
        shuffle(i)

    w2v = Word2Vec(x_w2v, size=size, window=window)

    X_train_w2v = copy.copy(X_train)
    X_test_w2v = copy.copy(X_test)

    for i in names:
        X_train_w2v[i] = X_train_w2v[i].astype('category')
        X_train_w2v[i].cat.categories = ['Feature %s %s' % (i, g) for g in X_train_w2v[i].cat.categories]

    for i in names:
        X_test_w2v[i] = X_test_w2v[i].astype('category')
        X_test_w2v[i].cat.categories = ['Feature %s %s' % (i, g) for g in X_test_w2v[i].cat.categories]

    X_train_w2v = X_train_w2v.values
    X_test_w2v = X_test_w2v.values

    x_w2v_train = np.random.random((len(X_train_w2v), size * X_train_w2v.shape[1]))
    x_w2v_test = np.random.random((len(X_test_w2v), size * X_test_w2v.shape[1]))

    for j in range(X_train_w2v.shape[1]):
        for i in range(X_train_w2v.shape[0]):
            if X_train_w2v[i, j] in w2v:
                x_w2v_train[i, j * size:(j + 1) * size] = w2v[X_train_w2v[i, j]]

    for j in range(X_test_w2v.shape[1]):
        for i in range(X_test_w2v.shape[0]):
            if X_test_w2v[i, j] in w2v:
                x_w2v_test[i, j * size:(j + 1) * size] = w2v[X_test_w2v[i, j]]

    l.fit(x_w2v_train, y_train)
    y_pred = l.predict_proba(x_w2v_test)
    print('{:.2f}'.format(log_loss(y_test, y_pred)))

    r.fit(x_w2v_train, y_train)
    y_pred = r.predict_proba(x_w2v_test)
    print('{:.2f}'.format(log_loss(y_test, y_pred)))


if __name__ == '__main__':
    main()
