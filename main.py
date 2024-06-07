import os
import jax
import time
import hydra
import dreamerv3
import ml_collections
import jax.numpy as jnp
import jax.random as random
from craftax.craftax_env import make_craftax_env_from_name
from craftax_wrapper import BatchEnvWrapper, OptimisticResetVecEnvWrapper


def make_dmc_env(env_name: str, use_egl=False, support_gpu=False, **kwargs):
    if use_egl and support_gpu:
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["MUJOCO_RENDERER"] = "egl"
    from mjx_envs import VecDmEnvWrapper, Walker2d, Cheetah
    if env_name == "walker2d":
        env = Walker2d()
    elif env_name == "cheetah":
        env = Cheetah()
    else:
        raise ValueError(f"Unsupported environment: {env_name}")
    env = VecDmEnvWrapper(env, **kwargs)

    return env



def make_craftax_env(env_name: str, autoreset: bool, num_envs: int=1):
    assert num_envs > 1, "number of the environments must be bigger than 1"
    env = make_craftax_env_from_name(env_name, not autoreset)
    if num_envs > 1:
        if autoreset:
            env = OptimisticResetVecEnvWrapper(env,
                                         num_envs=num_envs,
                                         reset_ratio=16) # fixed value; from the craftax paper
        else:
            env = BatchEnvWrapper(env)
    
    return env

def make_dreamer(env, config, key):
    obs_space = env.observation_space(env.default_params).shape
    act_space = env.action_space(env.default_params).n
    dreamer = dreamerv3.DreamerV3(key, obs_space, act_space, config=config)
    return dreamer


def craftax_rollout_fn(key, env, agent, agent_state, rollout_num=1000):
    def step_fn(carry, _):
        key, policy_key, step_key = random.split(carry["key"], num=3)
        policy_state, outs = agent.policy(policy_key, carry["policy_state"], carry)
        obs, env_state, reward, done, info = env.step(step_key, carry["env_state"], outs["action"].argmax(axis=1), env_params)
        return {"key": key, "env_state": env_state, "policy_state": policy_state, "observation": obs, "is_first": done}, {"observation": carry["observation"], "action": outs["action"], "reward": reward, "done": done, "info": info}
    rng, reset_key = jax.random.split(key)
    env_params = env.default_params
    first_obs, env_state = env.reset(reset_key, env_params)
    states = {"key": rng, "env_state": env_state, "policy_state": agent_state, "observation": first_obs, "is_first": jnp.bool_(jnp.ones((env.num_envs,)))}
    carry, outs = jax.lax.scan(step_fn, states, jnp.arange(rollout_num), unroll=False)
    return carry, outs


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    config = ml_collections.ConfigDict(cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    key = random.key(config.seed)
    print(f"Environment is compiling...")
    env = make_craftax_env(**config.env)
    print(f"Compiling is done...")
    dreamer = make_dreamer(env, config, key)
    dreamer_state = dreamer.policy_initial(config['env']['num_envs'])

    for epoch in range(config.num_epoch):
        print(f"Epoch {epoch}")
        a = time.time()
        carry, outs = craftax_rollout_fn(jax.random.key(0), env, dreamer, dreamer_state, rollout_num=config.num_steps)
        b = time.time()
        print(f"fps: {(config.env.num_envs * config.num_steps) / (b - a)}, elapsed time: {b - a}")


if __name__ == "__main__":
    main()
