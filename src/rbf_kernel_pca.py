from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np


def rbf_kernel_pca(X, gamma, n_components):

    # calculate pairwise squared Euclidean distances in the MxN dimensional dataset.
    sq_dists = pdist(X, 'sqeuclidean')

    # convert pairwise distances into a square matrix.
    mat_sq_dists = squareform(sq_dists)

    # compute the symmetric kernel matrix.
    K = exp(-gamma * mat_sq_dists)

    # center the kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # obtaining eigenpairs from the centered kernel matrix
    eigvals, eigvecs = eigh(K)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]

    # collect the top k eigenvectors (projected samples)
    alphas = np.column_stack((eigvecs[:, i] for i in range(n_components)))

    # collect the corresponding eigenvalues
    lambdas = [eigvals[i] for i in range(n_components)]

    return alphas, lambdas
