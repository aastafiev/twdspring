from dataclasses import dataclass
from typing import Self

import numpy as np


@dataclass
class Spring:
    query_vector: np.ndarray
    epsilon: float
    distance_type: str = 'quadratic'
    use_z_norm: bool = False

    def __post_init__(self):
        if self.query_vector.ndim != 1:
            raise ValueError("Query vector must be 1-dimensional.")

        self.query_vector_z_norm = (self.query_vector - np.mean(self.query_vector)) / np.std(self.query_vector)
        self.D = np.full((self.query_vector.shape[0]+1, 1), np.inf, dtype=np.float64)
        self.S = np.ones_like(self.D, dtype=np.int64)

        self.t = 0
        self.d_min = np.inf
        self.t_start, self.t_end = self.t, self.t

    def distance(self, x: float) -> float:
        query_vector = self.query_vector_z_norm if self.use_z_norm else self.query_vector

        match self.distance_type:
            case 'quadratic':
                return (x - query_vector) ** 2
            case 'absolute':
                return np.abs(x - query_vector)
            case _:
                raise ValueError("Invalid distance type.")

    def update_tick(self) -> Self:
        self.t += 1
        return self

    def update_state(self, x: float) -> Self:
        new_column = np.hstack((0, self.distance(x)))[..., np.newaxis]
        self.D = np.hstack((self.D, new_column))
        self.S = np.hstack((self.S, np.zeros_like(new_column, dtype=np.int64)))
        self.S[0, -1] = self.t

        for i in range(1, self.D.shape[0]):
            sub_d = np.copy(self.D[i-1:i+1, -2:])
            sub_d[1,1] = np.inf
            d_best = sub_d.min()
            self.D[i, -1] = self.D[i, -1] + d_best
            match sub_d[0, 1] == d_best, sub_d[1, 0] == d_best, sub_d[0, 0] == d_best:
                case np.True_, *_:
                    self.S[i, -1] = self.S[i-1, -1]
                case _, np.True_, _:
                    self.S[i, -1] = self.S[i, -2]
                case _, _, np.True_:
                    self.S[i, -1] = self.S[i - 1, -2]
        return self

    def search_step(self, x: float):
        self.update_tick().update_state(x)

        if self.d_min <= self.epsilon:
            if ((self.D[:, -1] >= self.d_min) | (self.S[:, -1] > self.t_end))[1:].all():
                yield 'match', float(self.d_min), self.t_start, self.t_end, self.t
                self.d_min = np.inf
                self.D[1:, -1] = np.where(self.S[1:, -1] <= self.t_end, np.inf, self.D[1:, -1])
        if self.D[-1, -1] <= self.epsilon and self.D[-1, -1] < self.d_min:
            self.d_min = self.D[-1, -1]
            self.t_start, self.t_end = int(self.S[-1, -1]), self.t
            yield 'tracking', float(self.d_min), self.t_start, self.t_end, self.t

        self.D[0, -1] = np.inf
        self.D[:, -2] = self.D[:, -1]
        self.D = self.D[:, 0:self.D.shape[1] - 1]  # for column vector return
        self.S[:, -2] = self.S[:, -1]
        self.S = self.S[:, 0:self.S.shape[1] - 1]  # for column vector return
