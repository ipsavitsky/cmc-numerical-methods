import numpy as np
from gauss import gauss


def test_gauss():
    test_matrix = np.array([[2.0, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]])
    test_right_part = np.array([[8.0, -11.0, 3.0]]).T
    x = gauss(test_matrix, test_right_part)
    assert np.allclose(x, np.array([-4.0, 9.0, -7.0]))
