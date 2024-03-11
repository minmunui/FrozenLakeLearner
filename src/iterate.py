from stable_baselines3 import PPO

from input_iterate import iterate_input
from src.env import make_env
from src.model import make_model_name
from src.train import train_model


def load_map_name(map_dir: str):
    """
    This function is used to load the maps from the given directory
    :param map_dir: path
    :return: list of maps
    """
    import os
    return os.listdir(map_dir)

def iterate(
        map_dir: str,
        algorithm: str,
        model_target: str,
        model_name: str,
        hyperparameters: dict,
        log_target: str,
):
    """
    This function is used to iterate the training process over the given maps
    :param map_dir: path to the directory containing the maps
    :param algorithm: algorithm to use for training
    :param model_target: path to the directory to save the model
    :param model_name: name of the model to save
    :param hyperparameters: hyperparameters for the algorithm
    :param log_target: path to the directory to save the logs
    :return: None
    """
    map_names = load_map_name(map_dir)
    each_timesteps = hyperparameters.pop('total_timesteps')/len(map_names)

    # TODO : Iterate over the maps
    if algorithm == 'PPO':
        env = make_env(map_path=f"{map_dir}/{map_name}", PPO=algorithm == 'PPO', gui=False)
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_target, **hyperparameters)

    for map_name in map_names:
        # make environment
        # make model name
        model_name = make_model_name(hyperparameters)
        # train model
        train_model(
            env=env,
            algorithm=algorithm,
            model_name=model_name,
            hyperparameters=hyperparameters
        )