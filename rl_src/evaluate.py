from stable_baselines3.common.evaluation import evaluate_policy


def evaluate_model(model, env):
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    return mean_reward, std_reward


def print_evaluate(env, model):
    mean_reward, std_reward = evaluate_model(model, env)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")
