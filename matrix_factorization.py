import numpy as np


def LU_decomposition(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Factorize matrix X into two matrices L and U using Gaussian elimination.

    :param X: the input matrix to be factorized
    :type X: numpy.ndarray

    :return: a tuple containing the lower triangular matrix L and the upper triangular matrix U
    :rtype: tuple[np.ndarray, np.ndarray]
    """

    n = X.shape[0]
    L = np.eye(n)
    U = X.copy()

    for i in range(n):
        for j in range(i + 1, n):
            if U[i, i] == 0:
                raise ValueError("Zero pivot encountered. Factorization failed.")
            L[j, i] = U[j, i] / U[i, i]
            U[j, :] -= L[j, i] * U[i, :]

    return L, U


def QR_decomposition(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the QR decomposition of a matrix using the Householder transformation.

    :param matrix: The input matrix to decompose.
    :type matrix: numpy.ndarray

    :return: The orthogonal matrix and upper triangular matrix in the decomposition.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    m, n = X.shape
    Q = np.eye(m)
    R = X.copy()

    for j in range(n):
        # Apply Householder transformation to zero-out below diagonal elements
        x = R[j:, j]
        e = np.zeros_like(x)
        e[0] = np.sign(x[0])
        u = x + e * np.linalg.norm(x)
        v = u / np.linalg.norm(u)
        R[j:, :] -= 2 * np.outer(v, np.dot(v, R[j:, :]))
        Q[:, j:] -= 2 * np.outer(Q[:, j:], np.dot(Q[:, j:].T, v))

    return Q, R
