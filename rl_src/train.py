from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


def train_model(env=None, algorithm="PPO", dir_path: str = "", model_name: str = "new_model", hyperparameters: dict = None):
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
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"{dir_path}/tensorboard", **agent_hyperparameters)
        model.learn(total_timesteps=timesteps)
        model.save(f"{dir_path}/{model_name}")

        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

        print(f"Mean reward: {mean_reward} +/- {std_reward}")
        return None
    else:
        print("Invalid algorithm")
        return None