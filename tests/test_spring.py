from itertools import dropwhile

import numpy as np
import pytest

from springpy import Searcher, Spring


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

@pytest.mark.parametrize('use_z_norm', [False, True], ids=['No z-norm', 'With z-norm'])
def test_update_state_method(use_z_norm):
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
    spring = Spring(query_vector=query, epsilon=15, use_z_norm=use_z_norm)

    x = [5, 12, 6, 10, 6, 5, 13]
    for val in x:
        spring.update_state(spring.z_norm(val))
    
    np.testing.assert_equal(spring.D, etalon_d)
    np.testing.assert_equal(spring.S, etalon_s)


@pytest.mark.parametrize('use_z_norm', [False, True], ids=['No z-norm', 'With z-norm'])
def test_search_step(use_z_norm):
    match use_z_norm:
        case False:
            etalon = [
                Searcher('tracking', 14.0, 3, 4, 4),
                Searcher('tracking', 14.0, 3, 4, 5),
                Searcher('tracking', 6.0, 3, 6, 6),
                Searcher('tracking', 6.0, 3, 6, 7),
                Searcher('match', 6.0, 3, 6, 8),
            ]
            epsilon = 15
        case True:
            etalon = [
                Searcher(status='tracking', distance=0.38867348067295193, t_start=3, t_end=6, t=6),
                Searcher(status='match', distance=0.38867348067295193, t_start=3, t_end=6, t=7),
                Searcher(status='match', distance=0.38867348067295193, t_start=3, t_end=6, t=8),
            ]
            epsilon = 0.5

    query = np.array((11, 6, 9, 4))
    spring = Spring(query_vector=query, epsilon=epsilon, use_z_norm=use_z_norm)

    x = [5, 6, 12, 6, 10, 6, 5, 13]
    results = []
    search_gen = spring.search()
    next(search_gen)
    results = (search_gen.send(val) for val in x)

    assert etalon == list(dropwhile(lambda x: not x.status, results))
