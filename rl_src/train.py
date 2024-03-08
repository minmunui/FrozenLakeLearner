from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from input_train import train_input
from rl_src.make_env import make_env
from utils.process_IO import make_model_name, create_directory_if_not_exists


def train_model(env=None, algorithm="PPO", dir_path: str = "", model_name: str = "new_model", hyperparameters: dict = None, tensorboard_log: str = ""):
    """
    train the model using the given algorithm and env
    then save the model with the given name
    :param hyperparameters:
    :param dir_path: path to the directory to save the model
    :param env: gym environment
    :param algorithm: algorithm to use for training
    :param model_name: name of the model to save
    :return: None
    """

    timesteps = hyperparameters['total_timesteps']

    agent_hyperparameters = hyperparameters
    agent_hyperparameters.pop('total_timesteps')

    if algorithm == "PPO":
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log, **agent_hyperparameters)
        model.learn(total_timesteps=timesteps)
        model.save(f"{dir_path}/{model_name}")

        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

        print(f"Mean reward: {mean_reward} +/- {std_reward}")
        return None
    else:
        print("Invalid algorithm")
        return None

def train_command():
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

    if env_options['map_path'] == "":
        map_name = f"_{env_options['map_size']}X{env_options['map_size']}_random"
    else:
        map_name = env_options['map_path'].split('/')[-1].split('.')[0]

    dir_to_save = f"models/{env_options['algorithm']['name']}/{map_name}"
    # make log directory
    dir_to_log = f"logs/{env_options['algorithm']['name']}/{map_name}/{model_name}"

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

    print("directory to save model : ", dir_to_save)
    print("directory to save logs : ", dir_to_log)