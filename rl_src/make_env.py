from utils.utils import print_map


def make_env(map_path: str = None, random_size: int = 5, PPO: bool = False, render_mode: str = None):
    """
    This function is used to create the environment for the agent to interact with.
    :param map_path: path to the map file
    :param random_size: if map_path is None, then the map will be randomly generated of size random_size x random_size
    :param PPO: whether to use PPO or not
    :return: gym environment
    """
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv

    if map_path != '':
        map_for_env = load_map(map_path)
    else:
        map_for_env = get_random_map(size=random_size)

    print("detected map : ")
    print_map(map_for_env)

    env = gym.make('FrozenLake-v1', desc=map_for_env, map_name=None, is_slippery=False, render_mode=render_mode)
    if PPO:
        env = DummyVecEnv([lambda: env])
    return env


def load_map(map_path: str = '/custom_maps/_5X5/_5X5_empty.txt'):
    """
    This function is used to load the map from the file.
    :param map_path:
    :return: map as array<string>
    """

    with open(map_path, 'r') as f:
        string_map = f.read()

    map_result = []
    for line in string_map.split('\n'):
        map_result.append(line)

    return map_result


def get_random_map(size: int = 5):
    """
    This function is used to generate a random map of size x size
    :param size: size of the map
    :return: array<string> size x size
    """
    from gymnasium.envs.toy_text.frozen_lake import generate_random_map
    return generate_random_map(size=size, p=0.8)
