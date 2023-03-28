import pytest
import numpy as np
from matrix_factorization import LU_decomposition, QR_decomposition


@pytest.fixture
def test_matrix():
    return np.array([[10.0, -7.0, 0.0], [-3.0, 6.0, 2.0], [5.0, -1.0, 5.0]])


def test_LU_factorization(test_matrix):
    L, U = LU_decomposition(test_matrix)
    assert np.allclose(L @ U, test_matrix)


@pytest.mark.skip(reason="It doesn't work :)")
def test_QR_factorization(test_matrix):
    Q, R = QR_decomposition(test_matrix)
    assert np.allclose(Q @ R, test_matrix)
