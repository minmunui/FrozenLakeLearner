"""
This file is used to simulate the input from the user.
"""
from utils.utils import get_merge_dictionary

User_Simulate_Input = {
    'algorithm': 'A2C',
    'model_path': 'models/A2C/iterate_2024-03-21_08_28_29/A2C_All_20000.zip',
    'map_path': 'maps/_4X4_custom2.txt',
    'map_dir': 'test/A2C/success_20000',
    'render_fps': 1,
}

Default_Simulate_Input = {
    'algorithm': 'PPO',
    'model_path': 'models/PPO/_5X5_2024-03-11_15_07_50/tt_5000_lr_0.0001_bs_64_g_0.99_ns_2048_.zip',
    'map_path': 'maps/_5X5_empty_rotate.txt',
    'map_dir': 'test_result/DQN/success/',
}


def simulate_input():
    return get_merge_dictionary(Default_Simulate_Input, User_Simulate_Input)
