import numpy as np
from gauss import gauss

def test_gauss():
    test_matrix = np.array([[2., 1., -1.], [-3., -1., 2.], [-2., 1., 2.]])
    test_right_part = np.array([[8., -11., 3.]]).T
    x = gauss(test_matrix, test_right_part)
    assert np.allclose(x, np.array([-4., 9., -7.]))


if __name__ == "__main__":
    test_gauss()