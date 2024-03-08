import os


def iterate_train(map_dir_path: str, algorithm: dict, target_dir: str):
    """
    This function is used to iterate over the map directory and train the model for each map
    the 'algorithm' parameter is dict like below
    {
        'name': 'PPO',
        'hyperparameters': {
            'total_timesteps': 50000,
            'learning_rate': 0.0001,
            'batch_size': 64,
            'gamma': 0.99,
            'n_steps': 2048,
        }
    }
    :param map_dir_path: path to the directory containing the custom_maps
    :param algorithm: algorithm to use for training.
    :param target_dir: path to the directory to save the models
    :return:
    """
    maps = os.listdir(map_dir_path)
