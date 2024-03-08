"""
Description : This file contains the default input for the model, you can edit the input as per your requirements.
"""
from utils.utils import get_merge_dictionary

User_Train_Input = {
    'model_name': '',
    'algorithm': {
        'name': 'PPO',
        'hyperparameters': {
            'total_timesteps': 50000,
            'learning_rate': 0.0001,
            'batch_size': 64,
            'gamma': 0.99,
            'n_steps': 2048,
        }
    },
    'map_path': 'maps/_5x5_empty',
    'map_dir_path': 'maps/train', # for iterate.py
    'target_dir': '', # for iterate.py
    'tensorboard_log': '',
    'map_size': '5',
}

Default_Train_Input = {
    'model_name': '',
    'algorithm': {
        'name': 'PPO',
        'hyperparameters': {
            'total_timesteps': 50000,
            'learning_rate': 0.0001,
            'batch_size': 64,
            'gamma': 0.99,
            'n_steps': 2048,
        }
    },
    'map_path': 'maps/_5x5_empty',
    'map_size': '5',
}


def train_input():
    """
    This function is used to get the user input from the user
    you can modify the default input as per your requirements.
    If you want to use the default input then set option to _5X5_empty string '', you can use the default_input object
    :return: dictionary
    """
    return get_merge_dictionary(User_Train_Input, Default_Train_Input)
