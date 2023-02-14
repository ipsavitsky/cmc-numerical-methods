import numpy as np
from matrix_factorization import matrix_factorization

def test_matrix_factorization():
    test_matrix = np.array([[10., -7., 0.], [-3., 6., 2.], [5., -1., 5.]])
    L, U = matrix_factorization(test_matrix)
    assert np.allclose(L @ U, test_matrix)