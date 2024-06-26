import numpy as np
from gymnasium import spaces
from envs.boolean.FixedGridFrozenLake import FixedGridFrozenLake


class Fixed1DFrozenLakeDiscrete(FixedGridFrozenLake):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.observation_space = spaces.Dict({
            'current': spaces.MultiDiscrete([self.nrow, self.ncol]),
            'goal': spaces.MultiDiscrete([self.nrow, self.ncol]),
            'map': spaces.MultiDiscrete(np.array([2] * self.nrow * self.ncol)),
        })
        # goal은 desc에서 G의 위치
        self.map = [1] * self.nrow * self.ncol
        for i in range(self.nrow):
            for j in range(self.ncol):
                if self.desc[i][j] == b'H':
                    self.map[i * self.ncol + j] = 0
