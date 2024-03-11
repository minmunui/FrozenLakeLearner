import os

from utils.utils import simplify_key


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


def make_model_directory(input_object: dict):
    dir_path = f"models/${input_object['map_name']}/${input_object['algorithm']}"
    if not os.path.join(dir_path):
        os.makedirs(dir_path)


def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
