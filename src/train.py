from stable_baselines3 import PPO
from input_train import train_input
from src.env import make_env
from utils.process_IO import create_directory_if_not_exists, get_model_name, get_model_path, \
    get_log_path, get_map_name


def train_model(
        env=None,
        algorithm: str = "PPO",
        model_target: str = "",
        model_name: str = "new_model",
        hyperparameters: dict = None,
        log_target: str = "",
        save: bool = True
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
    :param save: save the model or not
    :return: trained model
    """

    timesteps = hyperparameters['total_timesteps']

    agent_hyperparameters = hyperparameters
    agent_hyperparameters.pop('total_timesteps')

    if algorithm == "PPO":
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_target, **agent_hyperparameters)

    else:
        print("Invalid algorithm")
        return None

    model.learn(total_timesteps=timesteps)
    if save:
        model.save(f"{model_target}/{model_name}")
    return model


def train_command():
    env_options = train_input()
    print("detected env options", env_options)
    algorithm = env_options['algorithm']['name']
    hyperparameters = env_options['algorithm']['hyperparameters']

    # make environment
    env = make_env(map_path=env_options['map_path'], PPO=algorithm == 'PPO', gui=False)

    # process model name and log name
    map_name = get_map_name(env_options['map_path'], env_options['map_size'])
    model_name = get_model_name(env_options['model_name'], hyperparameters)
    model_dir = get_model_path(algorithm, env_options['model_target'], map_name)
    log_dir = get_log_path(algorithm, env_options['log_target'], map_name)

    # if directory does not exist then create it
    create_directory_if_not_exists(model_dir)
    create_directory_if_not_exists(log_dir)

    train_model(env=env, algorithm=algorithm, model_target=model_dir, model_name=model_name,
                hyperparameters=hyperparameters, log_target=log_dir)
