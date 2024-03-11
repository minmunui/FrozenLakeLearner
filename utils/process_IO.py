import os

from src.model import trim_extension
from utils.utils import simplify_key, current_time_for_file


def make_model_name(hyperparameters: dict):
    """
    This function takes a dictionary of hyperparameters and returns a string
    :param hyperparameters:
    :return:
    """
    name = ""
    for key in hyperparameters:
        name += f"{simplify_key(key)}_{hyperparameters[key]}_"
    name = name + ".zip"
    return name


def get_model_name(model_name: str, hyperparameters: dict):
    if model_name == '':
        result = make_model_name(hyperparameters)
    else:
        result = model_name
        extension = result.split('.')[-1]
        if extension != 'zip':
            result = result + ".zip"
    return result


def get_map_name(map_path, map_size):
    if map_path == '':
        map_name = f"_{map_size}_{current_time_for_file()}"
    else:
        map_name = trim_extension(map_path)
    return map_name


def get_model_path(algorithm: str, model_target: str, map_name: str, iter_model_name: str = None):
    if model_target == '':
        if iter_model_name is None:
            dir_to_model = f"models/{algorithm}/{map_name}/"
        else:
            if iter_model_name == '':
                dir_to_model = f"models/{algorithm}/iterate_{current_time_for_file()}/"
            else:
                dir_to_model = f"models/{algorithm}/{iter_model_name}/"
    else:
        dir_to_model = model_target
    return dir_to_model


def get_iter_model_path(algorithm: str, model_target: str, iter_model_name: str = ''):
    if model_target == '':
        if iter_model_name == '':
            dir_to_model = f"models/{algorithm}/iterate_{current_time_for_file()}/"
        else:
            dir_to_model = f"models/{algorithm}/iterate_{iter_model_name}/"
    else:
        dir_to_model = model_target
    return dir_to_model


def get_log_path(algorithm: str, log_target: str, map_name: str, iter_model_name: str = None):
    if log_target == '':
        if iter_model_name is None:
            dir_to_log = f"logs/{algorithm}/{map_name}/"
        else:
            if iter_model_name == '':
                dir_to_log = f"logs/{algorithm}/iterate_{current_time_for_file()}/"
            else:
                dir_to_log = f"logs/{algorithm}/{iter_model_name}/"
    else:
        dir_to_log = log_target
    return dir_to_log


def load_map_name(map_dir: str):
    """
    This function is used to load the maps from the given directory
    :param map_dir: path
    :return: list of maps
    """
    import os
    return os.listdir(map_dir)


def make_model_directory(input_object: dict):
    dir_path = f"models/${input_object['map_name']}/${input_object['algorithm']}"
    if not os.path.join(dir_path):
        os.makedirs(dir_path)


def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
