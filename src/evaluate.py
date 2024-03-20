from stable_baselines3.common.evaluation import evaluate_policy

from src.env import make_env
from src.model import get_algorithm


def evaluate_model(model, env):
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    return mean_reward, std_reward


def print_evaluate(env, model):
    mean_reward, std_reward = evaluate_model(model, env)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")


def evaluate_command(gui_render: bool = False, option: dict = None):
    env_options = option
    print("detected env options", env_options)

    env = make_env(map_path=env_options['map_path'], PPO=env_options['algorithm'] == 'PPO', gui=gui_render)

    model_path = env_options['model_path']

    print("model path : ", model_path)
    loaded_model = get_algorithm(env_options['algorithm']).load(model_path)
    # evaluate model

    print_evaluate(env=env, model=loaded_model)

