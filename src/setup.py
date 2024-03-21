from envs.Fixed1DFrozenLake import Fixed1DFrozenLake
from envs.Fixed1DFrozenLakeV2 import Fixed1DFrozenLakeV2


def env_class(**kwargs):
    print("kwargs : ", kwargs)
    return Fixed1DFrozenLake(**kwargs)
