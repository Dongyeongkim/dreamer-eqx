from jax import random
from dreamerv3.replay import tree_stack


def craftax_eval_fn(agent_fn, env_fn, key, agent_modules, env_params):
    key, policy_key, env_key = random.split(key, num=3)
    env_state, obs = env_fn.reset(env_key, env_params)
    agent_state = agent_fn.policy_initial(agent_modules, 1)
    timesteps = []
    while True:
        key, policy_key, env_key = random.split(key, num=3)
        env_state, timestep = env_fn.step(
            env_key, env_state, agent_state[0][1].argmax(axis=1), env_params
        )
        timestep["deter"] = agent_state[0][0]["deter"]
        timestep["stoch"] = agent_state[0][0]["stoch"]
        timestep["action"] = agent_state[0][1]
        timesteps.append(timestep)
        if timestep["is_terminal"]:
            break
        agent_state, outs = agent_fn.policy(
            agent_modules, policy_key, agent_state, timestep
        )
    timesteps = tree_stack(timesteps)
    forbidden_key = ["deter", "stoch", "action", "is_first", "is_last", "discount"]
    removed = [timesteps.pop(k) for k in forbidden_key]
    obses = timesteps.pop("observation")
    eplen = len(timesteps["score"])
    score_and_achievements = {f"eval/{k}": v[-1] for k, v in timesteps.items()}
    score_and_achievements.update({"eval/policy": obses})
    score_and_achievements.update({"eval/episode_length": eplen})

    return key, score_and_achievements
