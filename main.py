import os
import jax
import tqdm
import hydra
import dreamerv3
import ml_collections
import jax.numpy as jnp
import jax.random as random
from craftax.craftax_env import make_craftax_env_from_name
from craftax_wrapper import (
    BatchEnvWrapper,
    OptimisticResetVecEnvWrapper,
    CraftaxWrapper,
)
from dreamerv3.replay import generate_replaybuffer, pushstep, defragmenter, sampler


def make_dmc_env(env_name: str, use_egl=False, support_gpu=False, **kwargs):
    if use_egl and support_gpu:
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["MUJOCO_RENDERER"] = "egl"
    from envs import VecDmEnvWrapper, Walker2d, Cheetah

    if env_name == "walker2d":
        env = Walker2d()
    elif env_name == "cheetah":
        env = Cheetah()
    else:
        raise ValueError(f"Unsupported environment: {env_name}")
    env = VecDmEnvWrapper(env, **kwargs)

    return env


def make_craftax_env(env_name: str, autoreset: bool, num_envs: int = 1):
    assert num_envs > 0, "number of the environments must be greater or equal than 1"
    env = make_craftax_env_from_name(env_name, not autoreset)
    if num_envs > 1:
        if autoreset:
            env = OptimisticResetVecEnvWrapper(
                env, num_envs=num_envs, reset_ratio=16
            )  # fixed value; from the craftax paper
        else:
            env = BatchEnvWrapper(env)

    env = CraftaxWrapper(env)

    return env


def make_dreamer(env, config, key):
    obs_space = env.observation_space(env.default_params).shape
    act_space = env.action_space(env.default_params).n
    dreamer = dreamerv3.DreamerV3(key, obs_space, act_space, config=config)
    return dreamer


def train_step_fn(carry, num_interaction_steps):
    if num_interaction_steps % carry["train_every"] == 0:
        buffer_state = carry["buffer_state"]
        return carry, None
    else:
        return carry, None


# rollout_fn
#   - interaction: agent, agent_modules(params), env, env_state, replaybuffer_state, other configs -> env_state, replaybuffer_state
#   - model training: agent, agent_modules(params), optimizer, optimizer_state, replaybuffer_state -> agent_modules(params), opt_state, metrics(loss, images, blah blah)


def interaction_fn(
    key, agent_fn, agent_modules, agent_state, env_fn, env_params, env_state, rb_state
):
    key, policy_key, env_key = random.split(key)
    env_state, timestep = env_fn.step(
        env_key, env_state, agent_state[0][1].argmax(axis=1), env_params
    )
    timestep["action"] = agent_state[0][1]
    agent_state, outs = agent_fn.policy(
        policy_key, agent_modules, agent_state, timestep
    )

    rb_state = pushstep(rb_state, timestep)
    return agent_state, env_state, rb_state


# 1. interaction (env.step -> rb pushing -> send state over carry...)
# 2. worldmodel + actor-critic learning ( rb sampling -> training)
# 3. report?


def rollout_fn(agent, env):
    pass


def inference_fn(agent, env, agent_modules, replaybuffer, **kwargs):
    pass


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    config = ml_collections.ConfigDict(cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    key = random.key(config.seed)
    print(f"Environment is compiling...")
    env = make_craftax_env(**config.env)
    print(f"Compiling is done...")
    env_params = env.default_params
    rng, reset_key = jax.random.split(key)
    print(f"Setup the environments...")
    first_obs, env_state = env.reset(reset_key, env_params)
    print(f"Setting the environments is now done...")
    print("Building the agent...")
    key, dreamer_key = jax.random.split(key)
    dreamer = make_dreamer(env, config, dreamer_key)
    dreamer_state = dreamer.policy_initial(config["env"]["num_envs"])
    print("Building the agent is now done...")
    print("ReplayBuffer generation")
    rb_state = generate_replaybuffer(
        buffer_size=1_000_000,
        desired_key_dim={},
        batch_size=16,
        batch_length=65,
        num_env=16,
    )
    interaction_fn(
        jax.random.key(0),
        dreamer,
        _,
        dreamer_state,
        env,
        env_params,
        env_state,
        rb_state,
    )


if __name__ == "__main__":
    main()
