import os
import jax
import tqdm
import hydra
import dreamerv3
import ml_collections
import jax.numpy as jnp
import jax.random as random
from craftax.craftax_env import make_craftax_env_from_name
from craftax_wrapper import BatchEnvWrapper, OptimisticResetVecEnvWrapper
from dreamerv3.replay import ReplayBuffer


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
    assert num_envs > 1, "number of the environments must be bigger than 1"
    env = make_craftax_env_from_name(env_name, not autoreset)
    if num_envs > 1:
        if autoreset:
            env = OptimisticResetVecEnvWrapper(
                env, num_envs=num_envs, reset_ratio=16
            )  # fixed value; from the craftax paper
        else:
            env = BatchEnvWrapper(env)

    return env


def make_dreamer(env, config, key):
    obs_space = env.observation_space(env.default_params).shape
    act_space = env.action_space(env.default_params).n
    dreamer = dreamerv3.DreamerV3(key, obs_space, act_space, config=config)
    return dreamer


def craftax_rollout_fn(env, agent, states, rollout_num):
    def step_fn(carry, _):
        key, policy_key, step_key = random.split(carry["key"], num=3)
        policy_state, outs = agent.policy(policy_key, carry["policy_state"], carry)
        obs, env_state, reward, done, info = env.step(
            step_key,
            carry["env_state"],
            outs["action"].argmax(axis=1),
            env.default_params,
        )
        return {
            "key": key,
            "env_state": env_state,
            "policy_state": policy_state,
            "observation": jax.image.resize(
                obs, (obs.shape[0], 64, 64, obs.shape[3]), method="nearest"
            ),
            "is_first": done,
            "reward": reward,
        }, {
            "observation": carry["observation"],
            "deter": policy_state[0][0]["deter"],
            "stoch": policy_state[0][0]["stoch"],
            "action": outs["action"],
            "reward": reward,
            "is_first": carry["is_first"],
            "is_last": done,
            "is_terminal": jnp.equal(info["discount"], 0)
        }

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
    states = {
        "key": rng,
        "env_state": env_state,
        "policy_state": dreamer_state,
        "observation": jax.image.resize(
            first_obs,
            (first_obs.shape[0], 64, 64, first_obs.shape[3]),
            method="nearest",
        ),
        "is_first": jnp.bool_(jnp.ones((env.num_envs,))),
        "reward": jnp.zeros((env.num_envs,)),
    }
    rb = ReplayBuffer(
        buffer_size=5_000_000,
        key_and_desired_dim={
            "observation": 4,
            "deter": 2,
            "stoch": 3,
            "action": 2,
            "reward": 1,
            "is_first": 1,
            "is_last": 1,
            "is_terminal": 1,
        },
        batch_size=16,
        chunk_size=config.num_steps,
    )
    import time
    a = time.time()
    states, outs = craftax_rollout_fn(
        env, dreamer, states, rollout_num=config.num_steps
    )
    print(f"fps: {config.num_steps * config.env.num_envs /(time.time()-a)}")
    print(outs.keys())
    rb.push(outs)
    print(rb.buffer.keys())
    total_losses = []
    for _ in tqdm.tqdm(range(config.env.num_envs*config.num_steps//config.replay_ratio)):
        key, sampling_key, training_key = jax.random.split(key, num=3)
        chunk = rb.sample(sampling_key)
        total_loss, _ = dreamer.train(training_key, dreamer.train_initial(config.batch_size), chunk)


if __name__ == "__main__":
    main()
