from envs.Fixed1DFrozenLake import Fixed1DFrozenLake


def env_class(**kwargs):
    print("kwargs", kwargs)
    return Fixed1DFrozenLake(**kwargs)
