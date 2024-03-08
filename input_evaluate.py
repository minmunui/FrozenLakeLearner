"""
Description : This file contains the default input for the model, you can edit the input as per your requirements.
"""
from utils.utils import get_merge_dictionary

User_Evaluate_Input = {
    'algorithm': 'PPO',
    'model_path': 'models/PPO/_5X5_random/tt_50000_lr_0.0001_bs_64_g_0.99_ns_2048_.zip',
    'map_path': 'maps/_5X5_normal',
}

Default_Evaluate_Input = {
    'algorithm': 'PPO',
    'model_path': 'models/PPO/_5X5_random/tt_50000_lr_0.0001_bs_64_g_0.99_ns_2048_.zip',
    'map_path': 'maps/_5X5_normal',
}


def evaluate_input():
    return get_merge_dictionary(User_Evaluate_Input, Default_Evaluate_Input)
