import numpy as np
import pytest

from springpy import Searcher


def pytest_configure():
    pytest.query = np.array((11, 6, 9, 4))
    pytest.etalons = {
        True: {
            'epsilon': 0.5,
            'searcher': [
                Searcher(status='tracking', distance=0.38867348067295193, t_start=3, t_end=6, t=6),
                Searcher(status='match', distance=0.38867348067295193, t_start=3, t_end=6, t=7),
                Searcher(status='match', distance=0.38867348067295193, t_start=3, t_end=6, t=8),
            ],
            'D': np.array([[np.inf, np.inf, np.inf, 0., 0., 0., 0., 0.],
                           [np.inf, np.inf, np.inf, 5.28938991, 0.58575279, 4.90037434, 4.93394741, 0.0994402],
                           [np.inf, np.inf, np.inf, 5.48556271, 1.7773619, 0.7130059, 0.84571724, 4.81830481],
                           [np.inf, np.inf, np.inf, 7.91007956, 1.77787102, 2.87654554, 2.89887268, 1.965341],
                           [np.inf, np.inf, np.inf, 8., 5.14285714, 1.92691011, 2.07016151, 10.46300986]]),
            'S': np.array([[1, 1, 2, 3, 4, 5, 6, 7],
                           [1, 0, 0, 3, 4, 5, 6, 7],
                           [1, 0, 0, 3, 4, 4, 4, 7],
                           [1, 0, 0, 3, 4, 4, 4, 4],
                           [1, 0, 0, 3, 4, 4, 4, 4]]),
        },
            
        False: {
            'epsilon': 15,
            'searcher': [
                Searcher('tracking', 14.0, 3, 4, 4),
                Searcher('tracking', 14.0, 3, 4, 5),
                Searcher('tracking', 6.0, 3, 6, 6),
                Searcher('tracking', 6.0, 3, 6, 7),
                Searcher('match', 6.0, 3, 6, 8),
            ],
            'D': np.array([
                [np.inf, 0, 0, 0, 0, 0, 0, 0],
                [np.inf, 36, 1, 25, 1, 25, 36, 4],
                [np.inf, 37, 37, 1, 17, 1, 2, 51],
                [np.inf, 53, 46, 10, 2, 10, 17, 18],
                [np.inf, 54, 110, 14, 38,  6, 7, 88],
            ], dtype=np.float64),
            'S': np.array([
                [1, 1, 2, 3, 4, 5, 6, 7],
                [1, 1, 2, 3, 4, 5, 6, 7],
                [1, 1, 2, 2, 4, 4, 4, 4],
                [1, 1, 2, 2, 2, 4, 4, 4],
                [1, 1, 2, 2, 2, 2, 2, 2],
            ], dtype=np.int64),
        },

    }
