"""
Description : This file contains the default input for the model, you can edit the input as per your requirements.
"""


def user_input():
    """
    This function is used to get the user input from the user
    you can modify the default input as per your requirements.
    If you want to use the default input then set option to _5X5_empty string '', you can use the default_input object
    :return: dictionary
    """
    return {
        'model_name': '',
        'algorithm': 'PPO',
        'map_path': '',
        'map_size': '5',
        'hyperparameters': {
            'total_timestep': 50000,
            'learning_rate': 0.0001
        }
    }


default_input = {
    'model_name': '',
    'algorithm': 'PPO',
    'map_path': None,
    'map_size': '5*5',
    'total_timestep': 10000,
    'learning_rate': 0.0001
}


def make_input_object():
    required_input = default_input.keys()
    result = {}
    for key in required_input:
        if key in user_input:
            result[key] = user_input[key]
        else:
            result[key] = default_input[key]
    return result
