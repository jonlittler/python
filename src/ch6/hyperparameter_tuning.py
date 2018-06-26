import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline


def main():

    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',
                     header=None)

    X = df.iloc[:, 2:].values
    y = df.iloc[:, 1].values

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

    # grid search CV
    pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))

    param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    param_grid = [{'svc__C': param_range, 'svc__kernel': ['linear']},
                  {'svc__C': param_range, 'svc__gamma': param_range, 'svc__kernel': ['rbf']}]

    gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1)

    gs = gs.fit(X_train, y_train)
    print(gs.best_score_)
    print(gs.best_params_)

    # test dataset
    clf = gs.best_estimator_
    clf.fit(X_train, y_train)
    print('Test accuracy: {:.3f}'.format(clf.score(X_test, y_test)))

    # nested cross validation (svm vs. decision tree)
    # these are 5x2 cross validation
    # GridSearchCV = 2-fold
    # cross_val_score = 5-fold
    gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='accuracy', cv=2)
    scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
    print('SVM CV accuracy: {:.3f} +/- {:.3f}'.format(np.mean(scores), np.std(scores)))

    gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),
                      param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}], scoring='accuracy', cv=2)
    scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
    print('Decision Tree CV accuracy: {:.3f} +/- {:.3f}'.format(np.mean(scores), np.std(scores)))


if __name__ == '__main':
    main()
