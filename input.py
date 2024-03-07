"""
Description : This file contains the default input for the model, you can edit the input as per your requirements.
"""


def train_input():
    """
    This function is used to get the user input from the user
    you can modify the default input as per your requirements.
    If you want to use the default input then set option to _5X5_empty string '', you can use the default_input object
    :return: dictionary
    """
    return {
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
        'map_path': '',
        'map_size': '5',
    }


Default_Train_Input = {
    'model_name': '',
    'algorithm': 'PPO',
    'map_path': None,
    'map_size': '5*5',
    'total_timesteps': 10000,
    'learning_rate': 0.0001
}


def evaluate_input():
    return {
        'algorithm': 'PPO',
        'model_path': 'models/PPO/_5X5_random/tt_50000_lr_0.0001_bs_64_g_0.99_ns_2048_.zip',
        'map_path': 'maps/_5X5_normal',
    }


def simulate_input():
    return {
        'algorithm': 'PPO',
        'model_path': 'models/PPO/_5X5_random/tt_50000_lr_0.0001_bs_64_g_0.99_ns_2048_.zip',
        'map_path': 'maps/_5X5_normal',
    }


def make_input_object():
    required_input = Default_Train_Input.keys()
    result = {}
    for key in required_input:
        if key in train_input:
            result[key] = train_input[key]
        else:
            result[key] = Default_Train_Input[key]
    return result
