import gymnasium as gym
import numpy as np
from gymnasium.vector.utils import spaces

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode="human")

action_mapper = {
    (1, 0, 0, 0): 0,
    (0, 1, 0, 0): 1,
    (0, 0, 1, 0): 2,
    (0, 0, 0, 1): 3
}


class CustomActionSpaceWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=int)

    def action(self, action):
        # 원래의 action을 새로운 action space에 맞게 수정합니다.
        # 예를 들어, 원래의 action space가 Discrete(4)였다면, 0과 1 사이의 action을 0과 3 사이의 action으로 변환할 수 있습니다.
        print("Action:", action)


env = CustomActionSpaceWrapper(env)
env.reset()

for i in range(10):
    action = env.action_space.sample()
    print("Original action:", type(action))

done = False
while not done:
    user_input = input("Enter 4 numbers (0 or 1) separated by space: ")
    action = [int(i) for i in user_input.split()]
    [observe, reward, done, info] = env.step(action)
    print("Observation:", observe)
    print("Reward:", reward)
    print("Done:", done)
    print("Info:", info)