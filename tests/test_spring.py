import numpy as np
import pytest

from src.spring import Spring


def test_post_init_invalid_query_vector():
    with pytest.raises(ValueError, match="Query vector must be 1-dimensional."):
        Spring(query_vector=np.array([[1, 2], [3, 4]]), epsilon=1)

def test_post_init_valid_query_vector():
    query_vector = np.array([1, 2, 3])
    spring = Spring(query_vector=query_vector, epsilon=1)
    
    assert spring.query_vector_z_norm is not None
    assert spring.D.shape == (4, 1)
    assert np.all(spring.D == np.inf)
    assert spring.S.shape == (4, 1)
    assert spring.t == 0

def test_update_state_method():
    etalon_d = np.array([
        [np.inf, 0, 0, 0, 0, 0, 0, 0],
        [np.inf, 36, 1, 25, 1, 25, 36, 4],
        [np.inf, 37, 37, 1, 17, 1, 2, 51],
        [np.inf, 53, 46, 10, 2, 10, 17, 18],
        [np.inf, 54, 110, 14, 38,  6, 7, 88],
    ], dtype=np.float64)
    etalon_s = np.array([
        [1, 1, 2, 3, 4, 5, 6, 7],
        [1, 1, 2, 3, 4, 5, 6, 7],
        [1, 1, 2, 2, 4, 4, 4, 4],
        [1, 1, 2, 2, 2, 4, 4, 4],
        [1, 1, 2, 2, 2, 2, 2, 2],
    ], dtype=np.int64)

    query = np.array((11, 6, 9, 4))
    spring = Spring(query_vector=query, epsilon=15)

    x = [5, 12, 6, 10, 6, 5, 13]
    for i in x:
        spring.update_tick().update_state(i)
    
    np.testing.assert_equal(spring.D, etalon_d)
    np.testing.assert_equal(spring.S, etalon_s)


def test_search_step():
    etalon = [('tracking', 14.0, 2, 3, 3), ('tracking', 6.0, 2, 5, 5), ('match', 6.0, 2, 5, 7)]

    query = np.array((11, 6, 9, 4))
    spring = Spring(query_vector=query, epsilon=15)
    
    x = [5, 12, 6, 10, 6, 5, 13]
    results = [
        (status, d_min, t_start, t_end, tick)
        for val in x for status, d_min, t_start, t_end, tick in spring.search_step(val)
    ]

    assert etalon == results
