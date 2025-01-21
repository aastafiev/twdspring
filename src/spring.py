from dataclasses import dataclass
from typing import Self

import numpy as np


@dataclass
class Spring:
    query_vector: np.ndarray
    distance_type: str = 'quadratic'
    use_z_norm: bool = False

    def __post_init__(self):
        if self.query_vector.ndim != 1:
            raise ValueError("Query vector must be 1-dimensional.")

        self.query_vector_z_norm = (self.query_vector - np.mean(self.query_vector)) / np.std(self.query_vector)
        self.D = np.full((self.query_vector.shape[0]+1, 1), np.inf, dtype=np.float64)
        self.S = np.empty((self.query_vector.shape[0], 1), dtype=np.int64)
        self.t = 0

    def distance(self, x: float) -> float:
        query_vector = self.query_vector_z_norm if self.use_z_norm else self.query_vector

        match self.distance_type:
            case 'quadratic':
                return (x - query_vector) ** 2
            case 'absolute':
                return np.abs(x - query_vector)
            case _:
                raise ValueError("Invalid distance type.")

    def dtw(self, x: float) -> Self:
        self.t += 1
        new_column = np.hstack((0, self.distance(x)))[..., np.newaxis]
        self.D = np.hstack((self.D, new_column))

        for i in range(1, self.D.shape[0]):
            sub_d = np.copy(self.D[i-1:i+1, -2:])
            sub_d[1,1] = np.inf
            self.D[i, -1] = self.D[i, -1] + sub_d.min()
        return self


