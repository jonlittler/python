import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from src.MajorityVoteClassifier import MajorityVoteClassifier

def main():

    iris = load_iris()
    X, y = iris.data[50:, [1, 2]], iris.target[50:]
    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1, stratify=y)

    # train 3 classifiers
    clf1 = LogisticRegression(penalty='l2', C=0.001, random_state=1)
    clf2 = DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=0)
    clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')

    pipe1 = Pipeline([['sc', StandardScaler()], ['clf', clf1]])
    pipe3 = Pipeline([['sc', StandardScaler()], ['clf', clf3]])

    clf_labels = ['Logistic Regression', 'Decision Tree', 'KNN']
    print('\n10-fold cross validation:')
    for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
        scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring='roc_auc')
        print('ROC AUC: {:.2f} (+/- {:.2f}) {}'.format(scores.mean(), scores.std(), label))

    mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])
    clf_labels += ['Majority Voting']
    print('\n10-fold cross validation:')
    for clf, label in zip([pipe1, clf2, pipe3, mv_clf], clf_labels):
        scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring='roc_auc')
        print('ROC AUC: {:.2f} (+/- {:.2f}) {}'.format(scores.mean(), scores.std(), label))






if __name__ == '__main__':
    main()