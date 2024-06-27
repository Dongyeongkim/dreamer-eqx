import jax
import numpy as np
from jax import random
import jax.numpy as jnp
from dreamerv3.replay import tree_stack


def craftax_eval_fn(agent_fn, env_fn, key, agent_modules, env_params):
    key, policy_key, env_key = random.split(key, num=3)
    env_state, obs = env_fn.reset(env_key, env_params)
    agent_state = agent_fn.policy_initial(agent_modules, 1)
    timesteps = []
    timestep = {"is_terminal": False}
    while not timestep["is_terminal"]:
        key, policy_key, env_key = random.split(key, num=3)
        env_state, timestep = env_fn.step(
            env_key, env_state, agent_state[0][1].argmax(axis=1), env_params
        )
        timestep["deter"] = agent_state[0][0]["deter"]
        timestep["stoch"] = agent_state[0][0]["stoch"]
        timestep["action"] = agent_state[0][1]
        timesteps.append(timestep)
        agent_state, outs = agent_fn.policy(
            agent_modules, policy_key, agent_state, timestep
        )
    timesteps = tree_stack(timesteps)
    forbidden_key = ["deter", "stoch", "action", "reward", "is_first", "is_last", "is_terminal", "discount"]
    removed = [timesteps.pop(k) for k in forbidden_key]
    obses = timesteps.pop("observation")
    eplen = len(timesteps["score"])
    score_and_achievements = {f"eval/{k}": jnp.squeeze(v[-1]) for k, v in timesteps.items()}
    score_and_achievements.update({"eval/policy": jnp.squeeze(obses)})
    score_and_achievements.update({"eval/episode_length": jnp.array(eplen)})
    score_and_achievements = {k: np.float32(v) for k, v in score_and_achievements.items()}
    return key, score_and_achievements
