def get_algorithm(algorithm: str):
    if algorithm == 'PPO':
        from stable_baselines3 import PPO
        return PPO
    elif algorithm == 'DQN':
        from stable_baselines3 import DQN
        return DQN
    elif algorithm == 'A2C':
        from stable_baselines3 import A2C
        return A2C
    elif algorithm == 'SAC':
        from stable_baselines3 import SAC
        return SAC
    elif algorithm == 'TD3':
        from stable_baselines3 import TD3
        return TD3
    elif algorithm == 'DDPG':
        from stable_baselines3 import DDPG
        return DDPG
    else:
        raise ValueError("Invalid algorithm")


def prune_hyperparameters(hyperparameters: dict, algorithm: str):
    """
    This function is used to remove the hyperparameters that are not required for the given algorithm
    :param hyperparameters:
    :param algorithm:
    :return: pruned hyperparameters
    """
    if algorithm == 'DQN':
        hyperparameters.pop('n_steps', None)
        print("n_steps removed from hyperparameters for DQN")
    elif algorithm == 'A2C':
        hyperparameters.pop('batch_size', None)
        print("batch_size removed from hyperparameters for A2C")
    elif algorithm == 'SAC':
        hyperparameters.pop('n_steps')
        hyperparameters.pop('batch_size', None)
        print("n_steps and batch_size removed from hyperparameters for SAC")
    elif algorithm == 'TD3':
        hyperparameters.pop('n_steps')
        hyperparameters.pop('batch_size', None)
        print("n_steps and batch_size removed from hyperparameters for TD3")
    elif algorithm == 'DDPG':
        hyperparameters.pop('n_steps')
        hyperparameters.pop('batch_size', None)
        print("n_steps and batch_size removed from hyperparameters for DDPG")
    return hyperparameters


def trim_extension(map_path: str):
    return map_path.split('/')[-1].split('.')[0]


def get_random_map_name(n_col, n_row):
    return f"_{n_col}X{n_row}_random"


def dir_to_save(algorithm: str, map_name: str):
    return f"models/{algorithm}/{map_name}"


def dir_to_log(algorithm: str, hyperparameters: dict, map_name: str, model_name: str):
    from utils.process_IO import make_model_name
    if not model_name:
        model_name = make_model_name(hyperparameters)
    return f"logs/{algorithm}/{map_name}/{model_name}"


def save_path(dir_path: str, model_name: str):
    return f"{dir_path}/{model_name}"
