from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from envs.make_env import make_env


def evaluate_model(model, env):
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    return mean_reward, std_reward


def print_evaluate(model_path: str, map_path: str = None, is_PPO: bool = False):
    env = make_env(map_path=map_path, PPO=True)
    if is_PPO:
        model = PPO.load("ppo_frozenlake")
    else:
        model = PPO.load(model_path)
    mean_reward, std_reward = evaluate_model(model, env)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")
