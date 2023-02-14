import numpy as np

def matrix_factorization(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Factorize matrix X into two matrices L and U using Gaussian elimination.
    written by ChatGPT

    :param X: the input matrix to be factorized
    :return: a tuple containing the lower triangular matrix L and the upper triangular matrix U
    """
    
    n = X.shape[0]
    L = np.eye(n)
    U = X.copy()
    
    for i in range(n):
        for j in range(i+1, n):
            if U[i,i] == 0:
                raise ValueError("Zero pivot encountered. Factorization failed.")
            L[j,i] = U[j,i] / U[i,i]
            U[j,:] -= L[j,i] * U[i,:]
    
    return L, U
