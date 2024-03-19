from gymnasium import spaces
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv


class FixedGridFrozenLake(FrozenLakeEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.observation_space = spaces.Dict({
            'current': spaces.MultiDiscrete([self.nrow, self.ncol]),
            'goal': spaces.MultiDiscrete([self.nrow, self.ncol]),
            'map': spaces.MultiBinary([self.nrow, self.ncol]),
        })
        # goal은 desc에서 G의 위치
        self.goal = self._find_goal()

        self.map = [[True] * self.ncol for _ in range(self.nrow)]
        print("init map:", self.map)
        print(self.desc)
        for i in range(self.nrow):
            for j in range(self.ncol):
                if self.desc[i][j] == b'H':
                    print('i:', i, 'j:', j)
                    self.map[i][j] = False
        print(self.map)

    def _find_goal(self):
        for i in range(self.nrow):
            for j in range(self.ncol):
                if self.desc[i][j] == b'G':
                    return i, j

    def step(self, a):
        obs, reward, done, truncated, info = super().step(a)
        # print('obs:', obs)
        # print('self.s', self.s)
        current = (self.s // self.ncol, self.s % self.ncol)
        return {'current': current, 'goal': self.goal, 'map': self.map}, reward, done, truncated, info

    def reset(self, **kwargs):
        super().reset(**kwargs)
        current = (self.s // self.ncol, self.s % self.ncol)
        return {'current': current, 'goal': self.goal, 'map': self.map}, {}
