from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from input_train import train_input
from src.env import make_env
from src.model import get_map_name
from utils.process_IO import make_model_name, create_directory_if_not_exists
from utils.utils import current_time_for_file


def train_model(
        env=None,
        algorithm="PPO",
        model_target: str = "",
        model_name: str = "new_model",
        hyperparameters: dict = None,
        log_target: str = ""
):
    """
    train the model using the given algorithm and env
    then save the model with the given name
    :param env: gym environment
    :param algorithm: algorithm to use for training
    :param model_target: path to the directory to save the model
    :param model_name: name of the model to save
    :param hyperparameters:
    :param log_target:
    :return: trained model
    """

    timesteps = hyperparameters['total_timesteps']

    agent_hyperparameters = hyperparameters
    agent_hyperparameters.pop('total_timesteps')

    if algorithm == "PPO":
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_target, **agent_hyperparameters)
        model.learn(total_timesteps=timesteps)
        model.save(f"{model_target}/{model_name}")

        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        print("==========Train completed==========")
        print(f"model_path : {model_target}/{model_name}")
        print(f"log_path : {log_target}")
        print(f"Mean reward: {mean_reward} +/- {std_reward}")
        return model
    else:
        print("Invalid algorithm")
        return None


def train_command():
    env_options = train_input()
    print("detected env options", env_options)
    algorithm = env_options['algorithm']['name']
    hyperparameters = env_options['algorithm']['hyperparameters']

    # make environment
    env = make_env(map_path=env_options['map_path'], PPO=algorithm == 'PPO', gui=False)

    # get map name
    if env_options['map_path'] == '':
        map_name = f"_{env_options['map_size']}_{current_time_for_file()}"
    else:
        map_name = get_map_name(env_options['map_path'])

    # make model name
    if env_options['model_name'] == '':
        model_name = make_model_name(hyperparameters)
    else:
        model_name = env_options['model_name']
        print(f"detected : {model_name}")
        extension = model_name.split('.')[-1]
        print(f"extension")
        if extension != 'zip':
            model_name = model_name + ".zip"
            print(f"model_name : {model_name}")

    # make model directory
    if env_options['model_target'] == '':
        dir_to_model = f"models/{algorithm}/{map_name}/"
    else:
        dir_to_model = env_options['model_target']

    # make log directory
    if env_options['log_target'] == '':
        dir_to_log = f"logs/{algorithm}/{map_name}/"
    else:
        dir_to_log = env_options['log_target']

    # if directory does not exist then create it
    create_directory_if_not_exists(dir_to_model)
    create_directory_if_not_exists(dir_to_log)

    train_model(env=env, algorithm=algorithm, model_target=dir_to_model, model_name=model_name,
                hyperparameters=hyperparameters, log_target=dir_to_log)

