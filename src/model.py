from stable_baselines3 import PPO

from utils.utils import simplify_key


def make_model(algorithm: str, agent_hyperparameters: dict, env, tensorboard_log: str):
    if algorithm == 'PPO':
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log, **agent_hyperparameters)
        return model


def make_model_name(hyperparameters: dict):
    name = ""
    for key in hyperparameters:
        name += f"{simplify_key(key)}_{hyperparameters[key]}_"
    return name


def get_map_name(map_path: str):
    return map_path.split('/')[-1].split('.')[0]


def get_random_map_name(n_col, n_row):
    return f"_{n_col}X{n_row}_random"


def dir_to_save(algorithm: str, map_name: str):
    return f"models/{algorithm}/{map_name}"


def dir_to_log(algorithm: str, hyperparameters: dict, map_name: str, model_name: str):
    if not model_name:
        model_name = make_model_name(hyperparameters)
    return f"logs/{algorithm}/{map_name}/{model_name}"


def save_path(dir_path: str, model_name: str):
    return f"{dir_path}/{model_name}"