"""
Description : This file contains the default input for the model, you can edit the input as per your requirements.
"""
user_input = {
    'model_name': '',
    'algorithm': 'PPO',
    'map_path': 'maps/_5X5/empty',
    'map_size': '5*5',
    'step': 10000,
    'learning_rate': 0.0001
}

default_input = {
    'model_name': '',
    'algorithm': 'PPO',
    'map_path': None,
    'map_size': '5*5',
    'step': 10000,
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
