"""
This file is the main file to run the training and evaluation of the model

Usage:
    python main.py train
    python main.py evaluate
    python main.py simulate
    python main.py iterate
    python main.py generate 4X4
"""

import sys

from input_evaluate import evaluate_input
from input_simulate import simulate_input

from src.evaluate import evaluate_command, simulate_command
from src.iterate import iterate_command
from src.train import train_command
from utils.generate_maps import generate_all_map


def main():
    command = sys.argv[1]
    if command == "train":
        train_command()

    elif command == "evaluate":
        evaluate_command(option=evaluate_input())

    elif command == "simulate":
        simulate_command(option=simulate_input())

    elif command == "iterate":
        iterate_command()

    elif command == "generate":
        [n_row, n_col] = sys.argv[2].split('X')
        print(f"Generating all possible maps of size {n_row}x{n_col}")
        generate_all_map(int(n_col), int(n_row))

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
