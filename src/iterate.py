import random

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.Fixed1DFrozenLake import Fixed1DFrozenLake
from input_iterate import iterate_input
from src.env import make_env, load_map
from utils.process_IO import get_model_name, get_log_path, load_map_name, get_model_path


def iterate(
        map_dir: str,
        sample_num: int,
        algorithm: str,
        model_target: str,
        model_name: str,
        hyperparameters: dict,
        log_target: str,
):
    """
    This function is used to iterate the training process over the given maps
    :param sample_num: number of maps to sample from the map_dir if 0 then all maps will be used
    :param map_dir: path to the directory containing the maps
    :param algorithm: algorithm to use for training
    :param model_target: path to the directory to save the model
    :param model_name: name of the model to save
    :param hyperparameters: hyperparameters for the algorithm
    :param log_target: path to the directory to save the logs
    :return: None
    """
    maps_to_iterate = load_map_name(map_dir)
    if sample_num != 0:
        maps_to_iterate = random.sample(maps_to_iterate, sample_num)
    each_timesteps = hyperparameters.pop('total_timesteps') / len(maps_to_iterate)

    log_target = get_log_path(algorithm, log_target, 'None', iter_model_name=model_name)

    init_map_path = f"{map_dir}/{maps_to_iterate[0]}"
    init_map = load_map(init_map_path)
    print("Initial map", init_map)

    if algorithm == 'PPO':
        init_env = DummyVecEnv(
            [lambda: Fixed1DFrozenLake(desc=init_map, map_name=None, is_slippery=False, render_mode='None')])
        model = PPO("MultiInputPolicy", init_env, verbose=1, tensorboard_log=log_target, **hyperparameters)
    else:
        init_env = gym.make('FrozenLake-v1', desc=init_map, map_name=None, is_slippery=False, render_mode=None)
        model = None  # TODO : Add other algorithms

    for map_name in maps_to_iterate:
        env = make_env(map_path=f"{map_dir}/{map_name}", PPO=algorithm == 'PPO', gui=False)
        print(f"Training on map {map_name}")
        model.set_env(env=env)
        model.learn(total_timesteps=each_timesteps)

    model_name = get_model_name(model_name, hyperparameters)
    model_target = get_model_path(algorithm, model_target, 'None', iter_model_name=model_target)
    model.save(f"{model_target}/{model_name}")

    return model


def iterate_command():
    iterate_option = iterate_input()
    print("detected iterate options", iterate_option)
    algorithm = iterate_option['algorithm']['name']
    hyperparameters = iterate_option['algorithm']['hyperparameters']

    if iterate_option['map_dir'] == '':
        map_dir = 'maps/train'
    else:
        map_dir = iterate_option['map_dir']

    model_name = iterate_option['model_name']
    model_target = iterate_option['model_target']
    log_target = iterate_option['log_target']

    iterate(
        map_dir=map_dir,
        algorithm=algorithm,
        sample_num=iterate_option['sample_map'],
        model_target=model_target,
        model_name=model_name,
        hyperparameters=hyperparameters,
        log_target=log_target
    )
