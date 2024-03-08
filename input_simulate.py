"""
This file is used to simulate the input from the user.
"""
from utils.utils import get_merge_dictionary

User_Simulate_Input = {
    'algorithm': 'PPO',
    'model_path': 'models/PPO/_5x5_empty/tt_50000_lr_0.0001_bs_64_g_0.99_ns_2048_.zip',
    'map_path': 'maps/_5X5_empty_rotate',
}

Default_Simulate_Input = {
    'algorithm': 'PPO',
    'model_path': 'models/PPO/_5x5_empty/tt_50000_lr_0.0001_bs_64_g_0.99_ns_2048_.zip',
    'map_path': 'maps/_5X5_empty_rotate',
}


def simulate_input():
    return get_merge_dictionary(User_Simulate_Input, Default_Simulate_Input)
