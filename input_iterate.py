"""
Description : This file contains the default input for the model, you can edit the input as per your requirements.
"""
from utils.utils import get_merge_dictionary

User_Iterate_Input = {
    'model_name': '',  # name of the model to save
    'model_target': '',  # directory to save the model
    'map_dir': '',  # path to the directory containing the maps
    'log_target': '',  # path to the directory to save the logs
    'algorithm': {  # algorithm to use for training
        'name': 'PPO',
        'hyperparameters': {
            'total_timesteps': 500000,
            'learning_rate': 0.0001,
            'batch_size': 64,
            'gamma': 0.99,
            'n_steps': 2048,
        }
    },
}

Default_Iterate_Input = {
    'model_name': '',
    'model_target': '',
    'map_dir': '',
    'log_target': '',
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
}


def iterate_input():
    """
    This function is used to get the user input from the user
    you can modify the default input as per your requirements.
    If you want to use the default input then set option to _5X5_empty.txt string '', you can use the default_input object
    :return: dictionary
    """
    return get_merge_dictionary(Default_Iterate_Input, User_Iterate_Input)
