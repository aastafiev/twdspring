import numpy as np
import pytest

from src.spring import Spring


def test_post_init_invalid_query_vector():
    with pytest.raises(ValueError, match="Query vector must be 1-dimensional."):
        Spring(query_vector=np.array([[1, 2], [3, 4]]))

def test_post_init_valid_query_vector():
    query_vector = np.array([1, 2, 3])
    spring = Spring(query_vector=query_vector)
    
    assert spring.query_vector_z_norm is not None
    assert spring.D.shape == (4, 1)
    assert np.all(spring.D == np.inf)
    assert spring.S.shape == (3, 1)
    assert spring.t == 0

def test_dtw_method():
    etalon = np.array([
        [np.inf, 0, 0, 0, 0, 0, 0, 0],
        [np.inf, 36, 1, 25, 1, 25, 36, 4],
        [np.inf, 37, 37, 1, 17, 1, 2, 51],
        [np.inf, 53, 46, 10, 2, 10, 17, 18],
        [np.inf, 54, 110, 14, 38,  6, 7, 88],
 ], dtype=np.float64)

    query = np.array((11, 6, 9, 4))
    spring = Spring(query_vector=query)

    x = [5, 12, 6, 10, 6, 5, 13]
    for i in x:
        spring.dtw(i)
    
    np.testing.assert_equal(spring.D, etalon)
