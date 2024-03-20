from gymnasium import spaces
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv


class FixedGridFrozenLake(FrozenLakeEnv):
    def __init__(self, **kwargs):
        self.truncate = kwargs.get('truncate', False)
        kwargs.pop('truncate', None)
        super().__init__(**kwargs)

        self.observation_space = spaces.Dict({
            'current': spaces.MultiDiscrete([self.nrow, self.ncol]),
            'goal': spaces.MultiDiscrete([self.nrow, self.ncol]),
            'map': spaces.MultiBinary([self.nrow, self.ncol]),
        })
        # goal은 desc에서 G의 위치
        self.goal = self._find_goal()
        self.step_limit = self.ncol * self.nrow * 2
        self.n_step = 0

        self.map = [[True] * self.ncol for _ in range(self.nrow)]
        print(self.desc)
        for i in range(self.nrow):
            for j in range(self.ncol):
                if self.desc[i][j] == b'H':
                    self.map[i][j] = False

    def _find_goal(self):
        for i in range(self.nrow):
            for j in range(self.ncol):
                if self.desc[i][j] == b'G':
                    return i, j

    def step(self, a):
        obs, reward, done, truncated, info = super().step(a)
        current = (self.s // self.ncol, self.s % self.ncol)
        self.n_step += 1
        if self.truncate and self.n_step >= self.step_limit:
            done = True
            truncated = True
        return {'current': current, 'goal': self.goal, 'map': self.map}, reward, done, truncated, info

    def reset(self, **kwargs):
        super().reset(**kwargs)
        current = (self.s // self.ncol, self.s % self.ncol)
        return {'current': current, 'goal': self.goal, 'map': self.map}, {}
