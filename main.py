import os
import tqdm
from time import time
os.environ["MUJOCO_GL"] = "egl"
os.environ["MUJOCO_RENDERER"] = "egl"
from envs import VecDmEnvWrapper, Walker2d, Cheetah
from env_wrapper import DMC_JAX
import hydra
import dreamerv3
import ml_collections
from jax import random
import equinox as eqx

def make_env(env_name: str, **kwargs) -> VecDmEnvWrapper:
    if env_name == "walker2d":
        env = Walker2d()
    elif env_name == "cheetah":
        env = Cheetah()
    else:
        raise ValueError(f"Unsupported environment: {env_name}")
    env = VecDmEnvWrapper(env, **kwargs)
    env = DMC_JAX(env)
    
    return env


def make_dreamer(env, config, key):
    obs_space = env.observation_spec()
    act_space = env.action_spec()
    dreamer = dreamerv3.DreamerV3(key, obs_space, act_space, config=config)
    return dreamer


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    config = ml_collections.ConfigDict(cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    key = random.key(config.seed)
    print(f"Environment is compiling...")
    env = make_env(**config.env)
    print(f"Compiling is done...")
    dreamer = make_dreamer(env, config, key)
    dreamer_state = dreamer.policy_initial(config['env']['num_env'])

    for epoch in range(config.num_epoch):
        print(f"Epoch {epoch}")
        states = env.reset()
        a = time()
        while True:
            dreamer_state, outs = eqx.filter_jit(dreamer.policy)(key, dreamer_state, states)
            step_res = env.step(outs["action"])
            key, _ = random.split(key)
            if step_res["is_last"]:
                break
        b = time()
        print(f"fps: {(config.env.num_env*config.num_steps)/(b - a)}, elapsed time: {b - a}")



if __name__ == "__main__":
    main()
