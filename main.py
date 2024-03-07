import sys

from input import user_input
from rl_src.make_env import make_env
from rl_src.train import train_model
from utils.process_IO import make_model_name, create_directory_if_not_exists

command = ""

if len(sys.argv) > 1:
    command = sys.argv[1]

if command == "train":
    env_options = user_input()
    print("detected env options", env_options)

    # make environment
    env = make_env(map_path=env_options['map_path'], PPO=env_options['algorithm'] == 'PPO')

    # make model name
    if env_options['model_name'] == '' or 'model_name' not in env_options:
        model_name = make_model_name(env_options['hyperparameters'])
    else:
        model_name = env_options['model_name']

    print("model name : ", model_name)
    model_name = model_name + ".zip"

    # make model directory
    map_name = env_options['map_path'].split('/')[-1].split('.')[0]
    if env_options['map_path'] == "":
        dir_to_save = f"models/{env_options['algorithm']}/random_{env_options['map_size']}X{env_options['map_size']}"
    else:
        dir_to_save = f"models/{env_options['algorithm']}/{map_name}"
    print("directory to save : ", dir_to_save)

    # if directory does not exist then create it
    create_directory_if_not_exists(dir_to_save)

    # train model
    model = train_model(env=env,
                        algorithm=env_options['algorithm'],
                        dir_path=dir_to_save,
                        model_name=model_name,
                        total_timesteps=env_options['hyperparameters']['total_timestep']
                        )

if command == "evaluate":
    env_options = user_input()
    print("detected env options", env_options)
    env = make_env(map_path=env_options['map_path'], PPO=env_options['algorithm'] == 'PPO')
