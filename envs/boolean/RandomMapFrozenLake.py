import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map, LEFT, DOWN, RIGHT, UP

from env_setting import env_setting
from envs.boolean.Fixed1DFrozenLake import Fixed1DFrozenLake


class RandomMapFrozenLake(Fixed1DFrozenLake):
    def reset(self, **kwargs):
        self.desc = np.asarray(generate_random_map(size=self.ncol, p=env_setting()['frozen_prob']), dtype="c")
        self.goal = self._find_goal()
        self.start = self._find_start()
        self.current = self.start

        nA = 4
        nS = self.nrow * self.ncol

        self.P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row * self.ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, self.nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, self.ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)

        def update_probability_matrix(row, col, action):
            newrow, newcol = inc(row, col, action)
            newstate = to_s(newrow, newcol)
            newletter = self.desc[newrow, newcol]
            terminated = bytes(newletter) in b"GH"
            reward = float(newletter == b"G")
            return newstate, reward, terminated

        self.map = [True] * self.nrow * self.ncol

        for row in range(self.nrow):
            for col in range(self.ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = self.P[s][a]
                    letter = self.desc[row, col]
                    if letter in b"GH":
                        li.append((1.0, s, 0, True))
                        if letter == b'H':
                            self.map[row * self.ncol + col] = False
                    else:
                        li.append((1.0, *update_probability_matrix(row, col, a)))

        reset = super().reset(**kwargs)

        return reset

    def step(self, a):
        obs, reward, done, truncated, info = super().step(a)

        return obs, reward, done, truncated, info