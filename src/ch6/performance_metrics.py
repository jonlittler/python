import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import make_scorer


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

    # confusion matrix
    pipe_svc.fit(X_train, y_train)
    y_pred = pipe_svc.predict(X_test)
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print(confmat)

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.show()

    # precision, recall, f1-score
    print('Precision: {:.3f}'.format(precision_score(y_true=y_test, y_pred=y_pred)))
    print('Recall: {:.3f}'.format(recall_score(y_true=y_test, y_pred=y_pred)))
    print('F1: {:.3f}'.format(f1_score(y_true=y_test, y_pred=y_pred)))

    # make scorer
    scorer = make_scorer(f1_score, pos_label=0)
    gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring=scorer, cv=10)
    gs = gs.fit(X_train, y_train)
    print('Best Score: {:.3f}'.format(gs.best_score_))
    print('Best Params: {}'.format(gs.best_params_))


if __name__ == '__main':
    main()
