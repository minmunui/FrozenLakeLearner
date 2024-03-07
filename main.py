"""
This file is the main file to run the training and evaluation of the model

Usage:
    python main.py train
    python main.py evaluate
"""

import sys

from input import train_input, evaluate_input
from rl_src.make_env import make_env
from rl_src.train import train_model
from utils.process_IO import make_model_name, create_directory_if_not_exists


def main(command):
    if command == "train":
        env_options = train_input()
        print("detected env options", env_options)

        # make environment
        env = make_env(map_path=env_options['map_path'], PPO=env_options['algorithm']['name'] == 'PPO')

        # make model name
        if env_options['model_name'] == '' or 'model_name' not in env_options:
            model_name = make_model_name(env_options['algorithm']['hyperparameters'])
        else:
            model_name = env_options['model_name']

        print("model name : ", model_name)

        # make model directory
        map_name = env_options['map_path'].split('/')[-1].split('.')[0]
        if env_options['map_path'] == "":
            dir_to_save = f"models/{env_options['algorithm']['name']}/random_{env_options['map_size']}X{env_options['map_size']}"
        else:
            dir_to_save = f"models/{env_options['algorithm']['name']}/{map_name}"
        print("directory to save model : ", dir_to_save)

        # make log directory
        dir_to_log = f"logs/{env_options['algorithm']['name']}/{map_name}/{model_name}"
        print("directory to save logs : ", dir_to_log)

        # if directory does not exist then create it
        create_directory_if_not_exists(dir_to_save)
        create_directory_if_not_exists(dir_to_log)

        model_name = model_name + ".zip"
        # train model
        train_model(env=env,
                    algorithm=env_options['algorithm']['name'],
                    dir_path=dir_to_save,
                    model_name=model_name,
                    tensorboard_log=dir_to_log,
                    hyperparameters=env_options['algorithm']['hyperparameters']
                    )


    elif command == "evaluate":
        env_options = evaluate_input()
        print("detected env options", env_options)
        env = make_env(map_path=env_options['map_path'], PPO=env_options['algorithm'] == 'PPO')


    elif command == "simulate":
        pass

    else:
        print("Invalid command | Please use 'train', 'evaluate' or 'simulate' as command")
        return


if len(sys.argv) != 2:
    print("Invalid number of arguments")
    print("Usage: python main.py train")
    print("Usage: python main.py evaluate")
    print("Usage: python main.py simulate")

else:
    main(sys.argv[1])
