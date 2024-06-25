import tqdm
import jax
from jax import random
from dreamerv3.replay import defragmenter, pushstep, sampler

# rollout_fn
#   - interaction_fn: interacting with jax environment
#   - train_wm_fn: training wm with function


def train_and_evaluate_fn(
    key,
    num_steps,
    defrag_ratio,
    replay_ratio,
    logger,
    agent_fn,
    env_fn,
    opt_fn,
    agent_modules,
    agent_state,
    env_params,
    env_state,
    opt_state,
    rb_state,
):
    state = {
        "key": key,
        "agent_modules": agent_modules,
        "agent_state": agent_state,
        "opt_state": opt_state,
        "env_state": env_state,
        "rb_state": rb_state,
    }

    for idx in tqdm.tqdm(range(num_steps)):
        if idx % defrag_ratio == 0:
            state["key"], state["rb_state"] = defragmenter(
                state["key"], state["rb_state"], defrag_ratio, replay_ratio
            )

        if idx % replay_ratio == 0:
            state, lossval, loss_and_info = train_agent_fn(
                agent_fn,
                env_fn,
                opt_fn,
                defrag_ratio,
                replay_ratio,
                env_params=env_params,
                idx=idx,
                **state
            )
            if idx % 2 * replay_ratio == 0:
                logger._write(loss_and_info[1], env_fn.num_envs * idx)
        
        state = interaction_fn(agent_fn, env_fn, opt_fn, env_params=env_params, **state)
    return state


# prefill_fn
#   - interaction_fn: only interactions to fill replay buffer


def prefill_fn(
    key,
    num_steps,
    agent_fn,
    env_fn,
    opt_fn,
    agent_modules,
    agent_state,
    env_params,
    env_state,
    opt_state,
    rb_state,
):
    state = {
        "key": key,
        "agent_modules": agent_modules,
        "agent_state": agent_state,
        "opt_state": opt_state,
        "env_state": env_state,
        "rb_state": rb_state,
    }

    for _ in tqdm.tqdm(range(num_steps)):
        state = interaction_fn(agent_fn, env_fn, opt_fn, env_params=env_params, **state)

    return state["rb_state"]


def interaction_fn(
    agent_fn,
    env_fn,
    opt_fn,
    key,
    agent_modules,
    agent_state,
    env_params,
    env_state,
    opt_state,
    rb_state,
):
    key, policy_key, env_key = random.split(key, num=3)
    env_state, timestep = env_fn.step(
        env_key, env_state, agent_state[0][1].argmax(axis=1), env_params
    )
    timestep["deter"] = agent_state[0][0]["deter"]
    timestep["stoch"] = agent_state[0][0]["stoch"]
    timestep["action"] = agent_state[0][1]
    agent_state, outs = agent_fn.policy(
        agent_modules, policy_key, agent_state, timestep
    )

    rb_state = pushstep(rb_state, timestep)
    return {
        "key": key,
        "agent_modules": agent_modules,
        "agent_state": agent_state,
        "opt_state": opt_state,
        "env_state": env_state,
        "rb_state": rb_state,
    }


def train_agent_fn(
    agent_fn,
    env_fn,
    opt_fn,
    defrag_ratio,
    replay_ratio,
    key,
    agent_modules,
    agent_state,
    env_params,
    env_state,
    opt_state,
    rb_state,
    idx,
):
    key, training_key = random.split(key, num=2)
    sampled_data = sampler(
        idx,
        rb_state.cache,
        rb_state.deskeydim,
        rb_state.batch_size_dict,
        rb_state.batch_length_dict,
        defrag_ratio,
        replay_ratio,
    )
    agent_modules, opt_state, total_loss, loss_and_info = agent_fn.train(
        agent_modules,
        training_key,
        agent_state,
        sampled_data,
        opt_fn,
        opt_state,
    )
    return (
        {
            "key": key,
            "agent_modules": agent_modules,
            "agent_state": agent_state,
            "opt_state": opt_state,
            "env_state": env_state,
            "rb_state": rb_state,
        },
        total_loss,
        loss_and_info,
    )
