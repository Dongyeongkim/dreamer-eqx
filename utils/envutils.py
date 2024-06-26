import numpy as np

from craftax.craftax_env import make_craftax_env_from_name
from envs.wrappers.craftax_wrapper import (
    BatchEnvWrapper,
    OptimisticResetVecEnvWrapper,
    CraftaxWrapper,
)


def make_dmc_env(env_name: str, **kwargs):
    from envs import VecDmEnvWrapper, Walker2d, Cheetah

    if env_name == "walker2d":
        env = Walker2d()
    elif env_name == "cheetah":
        env = Cheetah()
    else:
        raise ValueError(f"Unsupported environment: {env_name}")
    env = VecDmEnvWrapper(env, **kwargs)

    return env


def make_craftax_env(env_name: str, autoreset: bool, num_envs: int = 1, **kwargs):
    assert num_envs > 0, "number of the environments must be greater or equal than 1"

    env = make_craftax_env_from_name(env_name, not autoreset)
    if autoreset:
        env = OptimisticResetVecEnvWrapper(
            env, num_envs=num_envs, reset_ratio=np.minimum(num_envs, 16)
        )
    else:
        env = BatchEnvWrapper(env)
    env = CraftaxWrapper(env)

    return env
