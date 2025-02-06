from itertools import dropwhile

import numpy as np
import pytest

from twdspring import Spring


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
    etalon_d = pytest.etalons[use_z_norm]['D']
    etalon_s = pytest.etalons[use_z_norm]['S']
    epsilon = pytest.etalons[use_z_norm]['epsilon']

    spring = Spring(query_vector=pytest.query, epsilon=epsilon, use_z_norm=use_z_norm)

    x = [5, 12, 6, 10, 6, 5, 13]
    for val in x:
        spring.update_tick().z_norm(val).update_state()
    
    np.testing.assert_allclose(spring.D, etalon_d)
    np.testing.assert_allclose(spring.S, etalon_s)


@pytest.mark.parametrize('use_z_norm', [False, True], ids=['No z-norm', 'With z-norm'])
def test_search(use_z_norm):
    etalon = pytest.etalons[use_z_norm]['searcher']
    epsilon = pytest.etalons[use_z_norm]['epsilon']

    spring = Spring(query_vector=pytest.query, epsilon=epsilon, use_z_norm=use_z_norm)

    x = [5, 6, 12, 6, 10, 6, 5, 13]
    results = (spring.step(val) for val in x)

    assert etalon == list(dropwhile(lambda x: not x.status, results))


def test_z_norm():
    spring = Spring(query_vector=pytest.query, epsilon=1, use_z_norm=True)
    x = [5, 6, 12, 6, 10, 6, 5, 13]
    x_z_norm = np.array([spring.update_tick().z_norm(val).current_x for val in x])
    spring.reset()

    x_z_norm_search = []
    for val in x:
        spring.step(val)
        x_z_norm_search.append(spring.current_x)

    np.testing.assert_allclose(x_z_norm, np.array(x_z_norm_search))


def test_search_z_norm():
    spring = Spring(query_vector=pytest.query, epsilon=.5, use_z_norm=True)
    x = [5, 6, 12, 6, 10, 6, 5, 13]
    etalon = list(dropwhile(lambda x: not x.status, (spring.step(val) for val in x)))

    spring.reset()
    query_vector_z_norm = (pytest.query - np.mean(pytest.query)) / np.std(pytest.query)
    spring.query_vector_z_norm = query_vector_z_norm
    z_norm_true = list(dropwhile(lambda x: not x.status, (spring.step(val) for val in x)))
    assert etalon == z_norm_true

    spring.reset()
    x_z_norm = np.array([spring.update_tick().z_norm(val).current_x for val in x])
    spring.reset()
    spring.use_z_norm = False
    spring.query_vector = query_vector_z_norm
    pre_z_norm = list(dropwhile(lambda x: not x.status, (spring.step(val) for val in x_z_norm)))
    assert etalon == pre_z_norm

    spring = Spring(query_vector=pytest.query, query_vector_z_norm=query_vector_z_norm, epsilon=.5, use_z_norm=True)
    z_norm_true = list(dropwhile(lambda x: not x.status, (spring.step(val) for val in x)))
    assert etalon == z_norm_true

    spring = Spring(query_vector=query_vector_z_norm, epsilon=.5, use_z_norm=False)
    pre_z_norm = list(dropwhile(lambda x: not x.status, (spring.step(val) for val in x_z_norm)))
    assert etalon == pre_z_norm
