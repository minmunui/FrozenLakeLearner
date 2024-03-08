from stable_baselines3.common.evaluation import evaluate_policy

from input_evaluate import evaluate_input
from rl_src.make_env import make_env


def evaluate_model(model, env):
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    return mean_reward, std_reward


def print_evaluate(env, model):
    mean_reward, std_reward = evaluate_model(model, env)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")


def evaluate_command(gui_render: bool = False, option: dict = None):
    env_options = option
    print("detected env options", env_options)

    render_mode = 'human' if gui_render else None

    env = make_env(map_path=env_options['map_path'], PPO=env_options['algorithm'] == 'PPO', render_mode=render_mode)

    model_path = env_options['model_path']

    print("model path : ", model_path)
    loaded_model = None

    if env_options['algorithm'] == 'PPO':
        from stable_baselines3 import PPO

        loaded_model = PPO.load(model_path)
    # evaluate model

    print_evaluate(env=env, model=loaded_model)
