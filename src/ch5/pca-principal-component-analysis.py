import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

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

    # covariance matrix
    cov_mat = np.cov(X_train_std.T)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    print('Eigenvalues:\n{}'.format(eigen_vals))

    tot = sum(eigen_vals)
    var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)

    plt.bar(range(1, 14), var_exp, alpha=0.5, align='center', label='individual explained variance')
    plt.step(range(1, 14), cum_var_exp, where='mid', label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    # Make a list of (eigenvalue, eigenvector) tuples
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eigen_pairs.sort(reverse=True)

    W = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
    print('Matrix W:\n', W)

    X_train_std[0].dot(W)
    X_train_pca = X_train_std.dot(W)

    colors = ['r', 'b', 'g']
    markers = ['s', 'x', 'o']
    for l, c, m in zip(np.unique(y_train), colors, markers):
        plt.scatter(X_train_pca[y_train == l, 0], X_train_pca[y_train == l, 1], c=c, label=l, marker=m)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.show()

    # now with scikit-learn
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)

    lr = LogisticRegression()
    lr.fit(X_train_pca, y_train)

    # train dataset
    plot_decision_regions(X_train_pca, y_train, lr)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.show()

    # test dataset
    plot_decision_regions(X_test_pca, y_test, lr)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.show()

    # explained variance
    pca = PCA(n_components=None)
    X_train_pca = pca.fit_transform(X_train_std)
    print('Explained Variance Ratio:\n{}'.format(pca.explained_variance_ratio_))

    # pca.components_ outputs an array of [n_components, n_features]
    # note: each coefficient represents the correlation between a particular pair of component and feature
    print(pd.DataFrame(pca.components_, columns=df_wine.iloc[:, 1:].columns))


if __name__ == '__main__':
    main()
