import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# 환경 생성
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
env = DummyVecEnv([lambda: env])  # PPO2 requires a vectorized environment to run

# 모델 초기화
model = PPO("MlpPolicy", env, verbose=1)

# 모델 학습
model.learn(total_timesteps=10000)

# 학습된 모델 테스트
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

print(f"Mean reward: {mean_reward} +/- {std_reward}")

# 학습된 모델 저장
model.save("ppo_frozenlake")
