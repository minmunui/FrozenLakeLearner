"""
Description : This file contains the default input for the model, you can edit the input as per your requirements.
"""
from utils.utils import get_merge_dictionary
import torch
User_Train_Input = {
    'model_name': 'A2C_128x3_60M',
    'model_target': 'network411/shrink',
    'map_path': 'maps/_4X4_simple.txt',
    'map_size': '',
    'log_target': 'network411/shrink/A2C_128x3_60M',
    'algorithm': {
        'name': 'A2C',
        'hyperparameters': {
            'total_timesteps': 60_000_000,
            'learning_rate': 0.0001,
            'batch_size': 64,
            'gamma': 0.99,
            'n_steps': 2048,
            'policy_kwargs': dict(
                # activation_fn=torch.nn.ReLU,
                net_arch=[128, 128, 128]
            )
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
            'total_timesteps': 400000,
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
