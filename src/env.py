from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

from utils.process_IO import create_directory_if_not_exists
from utils.utils import current_time_for_file


def make_env(map_path, gui: bool = False, env_class=None, truncate: bool = False, render_fps: int = 6):
    """
    This function is used to make the environment
    :param render_fps: fps for rendering the environment
    :param map_path: map to be used for the environment
    :param gui: if True then render the environment
    :param env_class:  class of the environment
    :param truncate: if True then use the truncated environment
    :return: environment
    """
    if map_path == "":
        map_for_env = get_random_map()
    else:
        map_for_env = load_map(map_path)

    if env_class is None:
        raise ValueError("env_class is None")
    render_mode = 'human' if gui else None
    env = env_class(desc=map_for_env, map_name=None, is_slippery=False, render_mode=render_mode, truncate=truncate, render_fps=render_fps)
    return env


def load_map(map_path: str):
    """
    This function is used to load the map from the file.
    :param map_path:
    :return: map as array<string>
    """
    with open(map_path, 'r') as f:
        string_map = f.read()
    map_result = []
    for line in string_map.split('\n'):
        if line != '':
            map_result.append(line)
    return map_result


def save_map(map_to_save: list):
    """
    This function is used to save the map to the file.
    :param map_to_save:
    :return: None
    """
    create_directory_if_not_exists('maps/generated')
    with open(f'maps/generated/g_{current_time_for_file()}.txt', 'w') as f:
        for line in map_to_save:
            f.write(line + '\n')


def get_random_map(size: int = 5):
    """
    This function is used to generate a random map of size x size
    :param size: size of the map
    :return: array<string> size x size
    """
    result = generate_random_map(size=size, p=0.8)
    save_map(result)
    return result
