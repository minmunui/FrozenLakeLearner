from envs.Fixed1DFrozenLake import Fixed1DFrozenLake


class Fixed1DFrozenLakeV2(Fixed1DFrozenLake):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.time_penalty = -1.0 / ((self.ncol + self.nrow) * 2)

    def step(self, action):
        prev_s = self.s
        obs, reward, done, truncated, info = super().step(action)

        current = (self.s // self.ncol, self.s % self.ncol)

        # 구멍일 경우 -0.8의 보상을 준다.
        if self.desc[current[0]][current[1]] == b'H':
            reward = -0.8
        # 이벤트가 없다면 해매는 시간에 대한 패널티를 준다.
        if not done:
            reward = self.time_penalty
        # 벽에 부딪힐 경우 -0.5의 보상을 준다.
        if prev_s == self.s:
            reward = -0.5

        return obs, reward, done, truncated, info
