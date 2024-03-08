import os

from rl_src.make_env import make_env
from utils.process_IO import make_model_name


def current_time():
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def train_in_maps(map_dir_path: str, algorithm: dict, target_dir: str, model_name: str = f"model_${current_time()}"):
    """
    This function is used to iterate over the map directory and train the model for each map
    the 'algorithm' parameter is dict like below
    {
        'name': 'PPO',
        'hyperparameters': {
            'total_timesteps': 50000,
            'learning_rate': 0.0001,
            'batch_size': 64,
            'gamma': 0.99,
            'n_steps': 2048,
        }
    }
    and save the model trained for all the maps in the map_dir_path to the target_dir
    :param map_dir_path: path to the directory containing the custom_maps
    :param algorithm: algorithm to use for training.
    :param target_dir: path to the directory to save the models
    :param model_name: name of the model to save
    :return: None
    """
    maps = os.listdir(map_dir_path)
    print("maps found in the directory : ")
    print(maps)

    for current_map in maps:
        print(f"training model for map : {current_map}")
        # make environment
        env = make_env(map_path=os.path.join(map_dir_path, current_map), PPO=algorithm['name'] == 'PPO')

        # make model directory
        map_name = current_map.split('/')[-1].split('.')[0]

        dir_to_save = os.path.join(target_dir, f"{algorithm['name']}/{map_name}")

        # make log directory
        dir_to_log = os.path.join(f"logs/{algorithm['name']}/{map_name}/{model_name}")

        # if directory does not exist then create it
        create_directory_if_not_exists(dir_to_save)
        create_directory_if_not_exists(dir_to_log)

        model_name = model_name + ".zip"
        # train model
        train_model(env=env,
                    algorithm=algorithm['name'],
                    dir_path=dir_to_save,
                    model_name=model_name,
                    tensorboard_log=dir_to_log,
                    hyperparameters=algorithm['hyperparameters']
                    )

        print("directory to save model : ", dir_to_save)
        print("directory to save logs : ", dir_to_log)
