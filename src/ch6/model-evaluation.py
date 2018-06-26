import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


def main():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',
                     header=None)

    X = df.iloc[:, 2:].values
    y = df.iloc[:, 1].values

    le = LabelEncoder()
    y = le.fit_transform(y)
    print(le.classes_)
    print(le.transform(['M', 'B']))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

    pipe_lr = make_pipeline(StandardScaler(), PCA(n_components=2), LogisticRegression(random_state=1))
    pipe_lr.fit(X_train, y_train)
    y_pred = pipe_lr.predict(X_test)
    print('Test Accuracy: {:.3f}'.format(pipe_lr.score(X_test, y_test)))

    # stratified k-fold
    kfold = StratifiedKFold(n_splits=10, random_state=1).split(X_train, y_train)

    scores = []
    for k, (train, test) in enumerate(kfold):
        pipe_lr.fit(X_train[train], y_train[train])
        score = pipe_lr.score(X_train[test], y_train[test])
        scores.append(score)
        print('Fold: {:2d}, Class distribution: {}, Accuracy: {:.3f}'.format((k+1), np.bincount(y_train[train]), score))

    print('\nCV Accuracy: {:.3f} +/- {:.3f}'.format(np.mean(scores), np.std(scores)))

    # scikit version
    scores = cross_val_score(estimator=pipe_lr, X=X_train, y=y_train, cv=10, n_jobs=-1)
    print('CV Accuracy Scores: {}'.format(scores))
    print('CV Accuracy: {:.3f} +/- {:.3f}'.format(np.mean(scores), np.std(scores)))


if __name__ == '__main':
    main()
