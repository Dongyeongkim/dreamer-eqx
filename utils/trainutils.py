import jax
import tqdm
import einops
from jax import random
import jax.numpy as jnp
from jax.tree import map as tree_map
from dreamerv3.replay import (
    put2buffer,
    put2fragmentcache,
    sampler,
    calcbufferidxes,
    calcfragmentidxes,
)

# rollout_fn
#   - interaction_fn: interacting with jax environment
#   - train_wm_fn: training wm with function


def train_and_evaluate_fn(
    key,
    num_steps,
    defrag_ratio,
    replay_ratio,
    debug_mode,
    report_ratio,
    eval_ratio,
    logger,
    agent_fn,
    env_fn,
    opt_fn,
    eval_fn,
    eval_env_fn,
    agent_modules,
    policy_state,
    imag_state,
    env_params,
    env_state,
    opt_state,
    rb_state,
):
    training_key, report_key, eval_key = random.split(key, num=3)
    state = {
        "key": training_key,
        "agent_modules": agent_modules,
        "policy_state": policy_state,
        "imag_state": imag_state, 
        "opt_state": opt_state,
        "env_state": env_state,
        "rb_state": rb_state,
    }
    is_online = False

    for idx in tqdm.tqdm(range(num_steps)):
        if idx % defrag_ratio == 0:
            is_online = True
            is_full, state["rb_state"].chunk_ptr, idxes = calcbufferidxes(
                state["rb_state"].chunk_ptr,
                state["rb_state"].num_chunks,
                state["rb_state"].num_env,
            )
            state["rb_state"].buffer = put2buffer(
                idxes, state["rb_state"].buffer, state["rb_state"].fragment
            )
            state["rb_state"].is_full = state["rb_state"].is_full or is_full

        if idx % replay_ratio == 0:
            state, lossval, loss_and_info = train_agent_fn(
                agent_fn,
                env_fn,
                opt_fn,
                defrag_ratio,
                replay_ratio,
                env_params=env_params,
                is_online=is_online,
                **state
            )
            is_online = False
            if debug_mode:
                logger._write(loss_and_info[2], env_fn.num_envs * idx)

        if idx % report_ratio == 0:
            report_key, report = report_fn(
                agent_fn,
                defrag_ratio,
                replay_ratio,
                report_key,
                state["agent_modules"],
                state["rb_state"],
                idx,
            )
            logger._write(report, env_fn.num_envs * idx)

        if idx % eval_ratio == 0:
            report = eval_fn(agent_fn, env_fn, key, agent_modules, env_params)
            logger._write(report, env_fn.num_envs * idx)

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
    policy_state,
    imag_state,
    env_params,
    env_state,
    opt_state,
    rb_state,
):
    state = {
        "key": key,
        "agent_modules": agent_modules,
        "policy_state": policy_state,
        "imag_state": imag_state,
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
    policy_state,
    imag_state,
    env_params,
    env_state,
    opt_state,
    rb_state,
):
    key, policy_key, env_key = random.split(key, num=3)
    env_state, timestep = env_fn.step(
        env_key, env_state, policy_state[0][1].argmax(axis=1), env_params
    )
    timestep["deter"] = policy_state[0][0]["deter"]
    timestep["stoch"] = jnp.argmax(policy_state[0][0]["stoch"], -1).astype(jnp.int32)
    timestep["action"] = policy_state[0][1]
    policy_state, outs = agent_fn.policy(
        agent_modules, policy_key, policy_state, timestep
    )
    rb_state.fragment = put2fragmentcache(
        rb_state.fragment_ptr, rb_state.fragment, timestep
    )
    rb_state.fragment_ptr = calcfragmentidxes(
        rb_state.fragment_ptr, rb_state.num_fragment
    )
    return {
        "key": key,
        "agent_modules": agent_modules,
        "policy_state": policy_state,
        "imag_state": imag_state,
        "opt_state": opt_state,
        "env_state": env_state,
        "rb_state": rb_state,
    }


def report_fn(agent_fn, defrag_ratio, replay_ratio, key, agent_modules, rb_state, idx):
    key, sampling_key, report_key = random.split(key, num=3)
    bufferlen = rb_state.num_chunks if rb_state.is_full else rb_state.chunk_ptr
    idx, sampled_data = sampler(
        sampling_key, bufferlen, rb_state.buffer, rb_state.batch_size
    )
    report = agent_fn.report(agent_modules, report_key, sampled_data)
    return key, report


def train_agent_fn(
    agent_fn,
    env_fn,
    opt_fn,
    defrag_ratio,
    replay_ratio,
    key,
    agent_modules,
    policy_state,
    imag_state,
    env_params,
    env_state,
    opt_state,
    rb_state,
    is_online,
):
    key, sampling_key, training_key = random.split(key, num=3)
    bufferlen = rb_state.num_chunks if rb_state.is_full else rb_state.chunk_ptr
    idx, sampled_data = sampler(
        sampling_key,
        bufferlen,
        rb_state.buffer,
        rb_state.batch_size,
        (
            jnp.arange(start=int(rb_state.chunk_ptr - 1), stop=int(rb_state.chunk_ptr))
            if is_online
            else None
        ),
    )
    agent_modules, opt_state, total_loss, loss_and_info = agent_fn.train(
        agent_modules,
        training_key,
        policy_state,
        sampled_data,
        opt_fn,
        opt_state,
    )
    replay_outs = loss_and_info[1]  # replay_outs
    deter = jnp.concatenate(
        (replay_outs["deter"], replay_outs["deter"][:, :1, ...]), axis=1
    )
    stoch = jnp.concatenate(
        (replay_outs["stoch"], replay_outs["stoch"][:, :1, ...]), axis=1
    )
    sampled_data["deter"] = deter
    sampled_data["stoch"] = stoch

    sampled_data = tree_map(
        lambda val: einops.rearrange(val, "b t ... -> (b t) 1 ..."), sampled_data
    )
    rb_state.buffer = put2buffer(
        idx, rb_state.buffer, sampled_data
    )  # because of the shape.
    return (
        {
            "key": key,
            "agent_modules": agent_modules,
            "policy_state": policy_state,
            "imag_state": loss_and_info[0],
            "opt_state": opt_state,
            "env_state": env_state,
            "rb_state": rb_state,
        },
        total_loss,
        loss_and_info,
    )
