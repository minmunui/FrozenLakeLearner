from env_setting import env_setting
from envs.boolean.Fixed1DFrozenLake import Fixed1DFrozenLake
from envs.boolean.FixedGridFrozenLake import FixedGridFrozenLake
from envs.boolean.RandomMapFrozenLake import RandomMapFrozenLake
from envs.discrete.Fixed1DFrozenLakeDiscrete import Fixed1DFrozenLakeDiscrete
from envs.discrete.RandomMapDiscreteShrinkDimension import RandomMapDiscreteShrinkDimension
from envs.discrete.RandomMapFrozenLakeDiscrete import RandomMapFrozenLakeDiscrete


def env_class(**kwargs):
    # print("kwargs : ", kwargs)
    env_params = env_setting()
#     print("env_class : ", env_params)
    if env_params["env_str"] == "1d":
        return Fixed1DFrozenLake(**kwargs, hole_penalty=env_params["hole_penalty"])
    elif env_params["env_str"] == "2d":
        return FixedGridFrozenLake(**kwargs, hole_penalty=env_params["hole_penalty"])
    elif env_params["env_str"] == "random":
        return RandomMapFrozenLake(**kwargs, hole_penalty=env_params["hole_penalty"])
    elif env_params["env_str"] == "discrete1d":
        return Fixed1DFrozenLakeDiscrete(**kwargs, hole_penalty=env_params["hole_penalty"])
    elif env_params["env_str"] == "discreteRandom":
        return RandomMapFrozenLakeDiscrete(**kwargs, hole_penalty=env_params["hole_penalty"])
    elif env_params["env_str"] == "discreteRandomShrink":
        return RandomMapDiscreteShrinkDimension(**kwargs, hole_penalty=env_params["hole_penalty"])
    else:
        raise ValueError(f"Invalid environment class : {str}")
