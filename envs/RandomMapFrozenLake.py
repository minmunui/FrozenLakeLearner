import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

from env_setting import env_setting
from envs.Fixed1DFrozenLake import Fixed1DFrozenLake


class RandomMapFrozenLake(Fixed1DFrozenLake):
    def reset(self, **kwargs):
        self.desc = np.asarray(generate_random_map(size=self.ncol, p=env_setting()['frozen_prob']), dtype="c")
        print(f"New map: {self.desc}")
        self.goal = self._find_goal()
        self.start = self._find_start()
        reset = super().reset(**kwargs)

        return reset
