"""
Description : This file contains the default input for the model, you can edit the input as per your requirements.
"""
from utils.utils import get_merge_dictionary

User_Train_Input = {
    'model_name': 'A2C_single_77000',
    'model_target': '',
    'map_path': 'maps/_4X4_custom2.txt',
    'map_size': '',
    'log_target': '',
    'algorithm': {
        'name': 'A2C',
        'hyperparameters': {
            'total_timesteps': 77000,
            'learning_rate': 0.0001,
            'batch_size': 64,
            'gamma': 0.99,
            'n_steps': 2048,
        }
    },
}

Default_Train_Input = {
    'model_name': '',
    'map_path': 'maps/_5x5_empty.txt',
    'map_size': '5X5',
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
    'log_target': '',
    'model_target': '',
}


def train_input():
    """
    This function is used to get the user input from the user
    you can modify the default input as per your requirements.
    If you want to use the default input then set option to _5X5_empty.txt string '', you can use the default_input object
    :return: dictionary
    """
    return get_merge_dictionary(Default_Train_Input, User_Train_Input)
