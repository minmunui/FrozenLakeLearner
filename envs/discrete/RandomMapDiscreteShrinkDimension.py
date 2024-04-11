from gymnasium import spaces

from envs.discrete.RandomMapFrozenLakeDiscrete import RandomMapFrozenLakeDiscrete


class RandomMapDiscreteShrinkDimension(RandomMapFrozenLakeDiscrete):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_space = spaces.MultiDiscrete([[4]*self.nrow * self.ncol])

    def reset(self, **kwargs):
        super().reset(**kwargs)
        print(self.map)
        print(self.current, self.goal)
        self.map[self.current[0] * self.ncol + self.current[1]] = 2
        self.map[self.goal[0] * self.ncol + self.goal[1]] = 3
        return self.map

    def step(self, a):
        print("prev", self.current)
        self.map[self.current[0] * self.ncol + self.current[1]] = 1
        obs, reward, done, truncated, info = super().step(a)
        self.map[self.current[0] * self.ncol + self.current[1]] = 2

        return self.map, reward, done, truncated, info
