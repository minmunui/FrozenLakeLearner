from time import sleep

import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

# 환경 생성
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode="human")
env = DummyVecEnv([lambda: env])  # PPO2 requires a vectorized environment to run


# 학습된 모델 불러오기
model = PPO.load("ppo_frozenlake")

# 화면에 렌더링, 키보드 입력을 통해 테스트
obs = env.reset()
done = False
env.render()

while not done:
    action, _states = model.predict(obs)
    env.render()
    obs, rewards, done, info = env.step(action)
    sleep(0.05)
    print("Action:", action, "Observation:", obs, "Reward:", rewards, "Done:", done, "Info:", info)
    if done:
        obs = env.reset()
        done = False