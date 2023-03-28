import numpy as np


def gauss(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve a linear system of equations Ax = b using Gaussian elimination.

    :param A: the coefficient matrix of the linear system.
    :type A: np.ndarray
    :param b: the right-hand side of the linear system.
    :type b: np.ndarray

    :return: The solution to the linear system.
    :rtype: numpy.ndarray
    """
    n = A.shape[0]

    # forward elimination
    for i in range(n):
        # pivot row search
        max_row = i + np.argmax(abs(A[i:, i]))
        # swap rows
        A[[i, max_row], :] = A[[max_row, i], :]
        b[[i, max_row]] = b[[max_row, i]]

        # elimination step
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            b[j] -= factor * b[i]
            A[j, i:] -= factor * A[i, i:]

    # back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1 :], x[i + 1 :])) / A[i, i]

    return x
