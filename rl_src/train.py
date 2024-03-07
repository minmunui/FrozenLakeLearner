from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


def train_model(env=None, algorithm="PPO", dir_path: str = "", model_name: str = "new_model",
                total_timesteps: int = 10000):
    """
    train the model using the given algorithm and env
    then save the model with the given name
    :param dir_path: path to the directory to save the model
    :param total_timesteps: number of timesteps to train the model
    :param env: gym environment
    :param algorithm: algorithm to use for training
    :param model_name: name of the model to save
    :return: None
    """
    if algorithm == "PPO":
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=total_timesteps)
        model.save(f"{dir_path}/{model_name}")

        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

        print(f"Mean reward: {mean_reward} +/- {std_reward}")
        return None
    else:
        print("Invalid algorithm")
        return None
