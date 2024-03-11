import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv


def SingleAgentEnv(map_for_env: list, gui=False):
    render_mode = 'human' if gui else None
    env = gym.make('FrozenLake-v1', desc=map_for_env, map_name=None, is_slippery=False, render_mode=render_mode)
    return env


def MultiAgentEnv(map_for_env: list, gui=False):
    env = DummyVecEnv([lambda: SingleAgentEnv(map_for_env, gui)])
    return env


def make_env(map_for_env, PPO: bool = True):
    """
    This function is used to make the environment
    :param map_for_env: map to be used for the environment
    :param PPO: if True then use PPO algorithm
    :return: environment
    """
    if PPO:
        env = MultiAgentEnv(map_for_env)
    else:
        env = SingleAgentEnv(map_for_env)
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
        map_result.append(line)
    return map_result
