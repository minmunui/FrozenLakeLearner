import shutil

from stable_baselines3.common.evaluation import evaluate_policy

from src.env import make_env
from src.model import get_algorithm
from src.setup import env_class
from utils.process_IO import load_map_name, create_directory_if_not_exists


def evaluate_model(model, env, n_eval_episodes=1):
    """
    This function is used to evaluate the model for the given environment
    :param model:
    :param env:
    :param n_eval_episodes:
    :return:
    """
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)
    return mean_reward, std_reward


def total_evaluation(model, map_dir: str, success_map_target: str = None):
    """
    This function is used to evaluate the model for the all the environments
    :param success_map_target: path to the directory to save the success maps
    :param model:
    :param map_dir:
    :param :
    :return:
    """
    map_names = load_map_name(map_dir)
    mean_rewards = 0.0
    n_success_input = 0
    n = len(map_names)
    for map_name in map_names:
        env = make_env(map_path=f"{map_dir}/{map_name}", gui=False, env_class=env_class, truncate=True)
        print(f"Evaluating on map {map_name}")
        mean_reward, _ = evaluate_model(model, env, n_eval_episodes=1)
        if mean_reward == 1:
            n_success_input += 1
            if success_map_target is not None:
                src_path = f"{map_dir}/{map_name}"
                dst_path = f"{success_map_target}/{map_name}"
                shutil.copy2(src_path, dst_path)

        mean_rewards += mean_reward

    return mean_rewards / n, n_success_input


def print_evaluate(env, model):
    mean_reward, std_reward = evaluate_model(model, env)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")


def simulate_command(option: dict = None):
    env_options = option
    print("detected env options", env_options)

    env = make_env(map_path=env_options['map_path'], gui=True,
                   env_class=env_class, render_fps=env_options['render_fps'])

    model_path = env_options['model_path']

    print("model path : ", model_path)

    loaded_model = get_algorithm(env_options['algorithm']).load(model_path)
    # evaluate model

    print_evaluate(env=env, model=loaded_model)


def iter_simulate_command(option: dict = None):
    env_options = option
    print("detected env options", env_options)

    map_dir = env_options['map_dir']
    map_names = load_map_name(env_options['map_dir'])

    for map_name in map_names:
        env = make_env(map_path=f"{map_dir}/{map_name}", gui=True,
                       env_class=env_class, render_fps=env_options['render_fps'])
        model_path = env_options['model_path']
        print("model path : ", model_path)
        loaded_model = get_algorithm(env_options['algorithm']).load(model_path)
        print(f"Simulating on map {map_name}")
        print_evaluate(env=env, model=loaded_model)


def evaluate_command(option: dict = None):
    env_options = option
    loaded_model = get_algorithm(env_options['algorithm']).load(env_options['model_path'])
    create_directory_if_not_exists(env_options['success_map_target'])
    result, n_success = total_evaluation(loaded_model,
                                         map_dir=env_options['map_dir'],
                                         success_map_target=env_options['success_map_target'])
    print("Evaluation complete")
    print("Model : ", env_options['model_path'])
    print(f"Mean reward : {result} | {n_success} / {len(load_map_name(env_options['map_dir']))}")
    if env_options['success_map_target'] is None or "":
        print("Success maps not saved")
    print("Success maps saved at : ", env_options['success_map_target'])
    return result
