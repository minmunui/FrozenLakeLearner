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

from src.evaluate import evaluate_command
from src.train import train_command


def main():
    command = sys.argv[1]
    if command == "train":
        train_command()

    elif command == "evaluate":
        evaluate_command(gui_render=False, option=evaluate_input())

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
