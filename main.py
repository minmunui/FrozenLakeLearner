"""
This file is the main file to run the training and evaluation of the model

Usage:
    python main.py train
    python main.py evaluate
    python main.py simulate
    python main.py train_in_maps
"""

import sys

from input_evaluate import evaluate_input
from input_simulate import simulate_input
from input_train import train_input

from rl_src.evaluate import evaluate_command
from rl_src.iterate import train_in_maps_command
from rl_src.train import train_command


def main():
    command = sys.argv[1]
    if command == "train":
        train_command()

    elif command == "evaluate":
        evaluate_command(gui_render=False, option=evaluate_input())

    elif command == "train_in_maps":
        train_in_maps_command()
        from rl_src.iterate import train_in_maps
        env_options = train_input()
        print("detected env options", env_options)

        train_in_maps(map_dir_path=env_options['map_path'], algorithm=env_options['algorithm'],
                      target_dir=env_options['target_dir'], model_name=env_options['model_name'])

    elif command == "simulate":
        evaluate_command(gui_render=True, option=simulate_input())

    else:
        print("Invalid command | Please use 'train', 'evaluate' or 'simulate' as command")
        return


if len(sys.argv) < 2:
    print("Invalid number of arguments")
    print("Usage: python main.py train")
    print("Usage: python main.py evaluate")
    print("Usage: python main.py simulate")

else:
    main()
