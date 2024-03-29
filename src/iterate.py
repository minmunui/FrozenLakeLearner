import random
from datetime import datetime

import gymnasium as gym

from src.env import make_env, load_map
from src.model import get_algorithm, prune_hyperparameters
from src.setup import env_class
from utils.process_IO import get_model_name, get_log_path, load_map_name, get_model_path


def iterate(
        map_dir: str,
        sample_num: int,
        algorithm: str,
        model_target: str,
        model_name: str,
        hyperparameters: dict,
        log_target: str,
        env: gym.Env
):
    """
    This function is used to iterate the training process over the given maps
    :param env: class of the environment
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

    init_env = env(desc=init_map, map_name=None, is_slippery=False, render_mode='None')
    alg_object = get_algorithm(algorithm)

    hyperparameters = prune_hyperparameters(hyperparameters, algorithm)

    model = alg_object("MultiInputPolicy", init_env, verbose=1, tensorboard_log=log_target, **hyperparameters)

    count = 0
    for map_name in maps_to_iterate:
        env = make_env(map_path=f"{map_dir}/{map_name}", gui=False, env_class=env_class,
                       truncate=False)
        print(f"Training on map {map_name} | {count + 1}/{len(maps_to_iterate)}")
        model.set_env(env=env)
        model.learn(total_timesteps=each_timesteps)
        count += 1

    model_name = get_model_name(model_name, hyperparameters)
    model_target = get_model_path(algorithm, model_target, 'None', iter_model_name=model_target)
    model.save(f"{model_target}/{model_name}")
    print(f"Training completed on {len(maps_to_iterate)} maps")
    print(f"Model saved at {model_target}/{model_name}")
    print(f"log saved at {log_target}")
    print(f"current time : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return model


def iterate_command(options: dict):
    print("detected iterate options", options)
    algorithm = options['algorithm']['name']
    hyperparameters = options['algorithm']['hyperparameters']

    if options['map_dir'] == '':
        map_dir = 'maps/train'
    else:
        map_dir = options['map_dir']

    model_name = options['model_name']
    model_target = options['model_target']
    log_target = options['log_target']

    iterate(
        map_dir=map_dir,
        algorithm=algorithm,
        sample_num=options['sample_map'],
        model_target=model_target,
        model_name=model_name,
        hyperparameters=hyperparameters,
        log_target=log_target,
        env=env_class
    )
