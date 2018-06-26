import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from src.plot_decision_regions import plot_decision_regions


def main():

    df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)

    df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
                       'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity',
                       'Hue', 'OD280/OD315 of diluted wines', 'Proline']

    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

    # standardize the features
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    # mean vecs
    np.set_printoptions(precision=4)
    mean_vecs = []
    for label in range(1, 4):
        mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))
        print('MV {}: {}\n'.format(label, mean_vecs[label-1]))

    # within-class scatter matrix
    d = 13
    S_W = np.zeros((d, d))
    for label, mv in zip(range(1, 4), mean_vecs):
        class_scatter = np.zeros((d, d))
        for row in X_train_std[y_train == label]:
            row, mv = row.reshape(d, 1), mv.reshape(d, 1)
            class_scatter += (row - mv).dot((row - mv).T)
        S_W += class_scatter
    print('Within-class scatter matrix: {}x{}'.format(S_W.shape[0], S_W.shape[1]))

    # are class labels in training set uniformly distributed?
    print('Class label distribution: {}'.format(np.bincount(y_train)[1:]))

    # scaled-within-class scatter matrix
    d = 13
    S_W = np.zeros((d, d))
    for label, mv in zip(range(1, 4), mean_vecs):
        class_scatter = np.cov(X_train_std[y_train == label].T)
        S_W += class_scatter
    print('Scaled-within-class scatter matrix: {}x{}'.format(S_W.shape[0], S_W.shape[1]))

    # between-class scatter matrix
    mean_overall = np.mean(X_train_std, axis=0)
    d = 13
    S_B = np.zeros((d, d))
    for i, mean_vec in enumerate(mean_vecs):
        n = X_train[y_train == i + 1].shape[0]
        mean_vec = mean_vec.reshape(d, 1)           # make column vector
        mean_overall = mean_overall.reshape(d, 1)
        S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
    print('Between-class scatter matrix: {}x{}'.format(S_B.shape[0], S_B.shape[1]))

    eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
    # eigen_pairs.sort(reverse=True)
    eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

    print('Eigenvalues in descending order:\n')
    for eigen_val in eigen_pairs:
        print('{:.4f}'.format(eigen_val[0]))

    # plot discriminibility
    tot = sum(eigen_vals.real)
    discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
    cum_discr = np.cumsum(discr)

    plt.bar(range(1, 14), discr, alpha=0.5, align='center', label='individual "discriminiability"')
    plt.step(range(1, 14), cum_discr, where='mid', label='cumulative "discriminibility"')
    plt.xlabel('Linear Discriminants')
    plt.ylabel('"Discriminibility" ratio')
    plt.ylim([-0.1, 1.1])
    plt.legend(loc='best')
    plt.show()

    W = np.hstack((eigen_pairs[0][1][:, np.newaxis].real, eigen_pairs[1][1][:, np.newaxis].real))
    print('Matrix W:\n', W)

    X_train_lda = X_train_std.dot(W)

    colors = ['r', 'b', 'g']
    markers = ['s', 'x', 'o']
    for l, c, m in zip(np.unique(y_train), colors, markers):
        plt.scatter(X_train_lda[y_train == l, 0], X_train_lda[y_train == l, 1], c=c, label=l, marker=m)
    plt.xlabel('LD 1')
    plt.ylabel('LD 2')
    plt.legend(loc='lower right')
    plt.show()

    # scikit-learn
    lda = LDA(n_components=2)
    X_train_lda = lda.fit_transform(X_train_std, y_train)
    X_test_lda = lda.transform(X_test_std)

    lr = LogisticRegression()
    lr.fit(X_train_lda, y_train)

    # train dataset
    plot_decision_regions(X_train_lda, y_train, lr)
    plt.xlabel('LD 1')
    plt.ylabel('LD 2')
    plt.legend(loc='lower left')
    plt.show()

    # test dataset
    plot_decision_regions(X_test_lda, y_test, lr)
    plt.xlabel('LD 1')
    plt.ylabel('LD 2')
    plt.legend(loc='lower left')
    plt.show()


if __name__ == '__main__':
    main()
