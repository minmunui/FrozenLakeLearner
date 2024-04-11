from gymnasium import spaces

from envs.discrete.RandomMapFrozenLakeDiscrete import RandomMapFrozenLakeDiscrete


class RandomMapDiscreteShrinkDimension(RandomMapFrozenLakeDiscrete):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_space = spaces.Dict({"map": spaces.MultiDiscrete([4] * self.nrow * self.ncol)})
        print(self.observation_space)

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self.map[self.current[0] * self.ncol + self.current[1]] = 2
        self.map[self.goal[0] * self.ncol + self.goal[1]] = 3
        return self.map

    def step(self, a):
        self.map[self.current[0] * self.ncol + self.current[1]] = 1
        obs, reward, done, truncated, info = super().step(a)
        self.map[self.current[0] * self.ncol + self.current[1]] = 2
        obs = {"map": self.map}

        return obs, reward, done, truncated, info
