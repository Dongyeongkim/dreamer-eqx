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
    prev = 0
    for idx in tqdm.tqdm(range(num_steps)):
        if (idx + 1) % defrag_ratio == 0:
            is_online = True
            is_full, state["rb_state"].buffer_ptr, idxes = calcbufferidxes(
                state["rb_state"].buffer_ptr,
                state["rb_state"].bufferlen_per_env,
                state["rb_state"].batch_length,
                )
            state["rb_state"].buffer = put2buffer(
                idxes, state["rb_state"].buffer, state["rb_state"].fragment
                )
            state["rb_state"].is_full = state["rb_state"].is_full or is_full

        if idx % 10 == 0:
            if idx == 0:
                repeats = 1
            else:
                repeats = int((idx - prev) * replay_ratio)
                prev += repeats / replay_ratio
            state = train_agent_fn(
                idx,
                agent_fn,
                env_fn,
                opt_fn,
                logger,
                defrag_ratio,
                replay_ratio,
                env_params=env_params,
                is_online=is_online,
                train_steps=repeats,
                debug=debug_mode,
                **state
            )
            is_online = False

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

    for i in tqdm.tqdm(range(num_steps)):
        state = interaction_fn(agent_fn, env_fn, opt_fn, env_params=env_params, **state)
        if (i + 1) % state["rb_state"].batch_length == 0:
            is_full, nextbuffer_ptr, idxes = calcbufferidxes(
                state["rb_state"].buffer_ptr,
                state["rb_state"].bufferlen_per_env,
                state["rb_state"].batch_length,
            )
            state["rb_state"].buffer_ptr = nextbuffer_ptr
            state["rb_state"].is_full = state["rb_state"].is_full or is_full
            state["rb_state"].buffer = put2buffer(idxes, state["rb_state"].buffer, state["rb_state"].fragment)

    return state


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
    timestep["stoch"] = policy_state[0][0]["stoch"]
    timestep["action"] = policy_state[0][1]
    policy_state, outs = agent_fn.policy(
        agent_modules, policy_key, policy_state, timestep
    )
    timestep["stoch"] = jnp.argmax(timestep["stoch"], -1).astype(jnp.int32)
    rb_state.fragment = put2fragmentcache(
        rb_state.fragment_ptr, rb_state.fragment, timestep
    )
    rb_state.fragment_ptr = calcfragmentidxes(
        rb_state.fragment_ptr, rb_state.batch_length
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
    bufferlen = rb_state.bufferlen_per_env if rb_state.is_full else rb_state.buffer_ptr
    _, _, _, sampled_data = sampler(
        sampling_key,
        rb_state.buffer,
        rb_state.num_env,
        rb_state.buffer_ptr,
        rb_state.online_ptr,
        bufferlen,
        rb_state.batch_size,
        rb_state.batch_length,
    )
    report = agent_fn.report(agent_modules, report_key, sampled_data)
    return key, report


def train_agent_fn(
    idx,
    agent_fn,
    env_fn,
    opt_fn,
    logger,
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
    train_steps=1,
    debug=True,
):
    loss_and_info = None
    learning_state = agent_modules["wm"].initial(16)
    for i in reversed(range(train_steps)):
        key, sampling_key, training_key = random.split(key, num=3)
        bufferlen = rb_state.bufferlen_per_env if rb_state.is_full else rb_state.buffer_ptr
        env_idxes, timestep_idxes, rb_state.online_ptr, sampled_data = sampler(
            sampling_key,
            rb_state.buffer,
            rb_state.num_env,
            rb_state.buffer_ptr,
            rb_state.online_ptr,
            bufferlen,
            rb_state.batch_size,
            rb_state.batch_length,
        )

        agent_modules, opt_state, total_loss, loss_and_info = agent_fn.train(
            agent_modules,
            training_key,
            learning_state,
            sampled_data,
            opt_fn,
            opt_state,
        )
        replay_outs = loss_and_info[1]  # replay_outs
        deter = jnp.concatenate(
            (sampled_data["deter"][:, :1, ...], replay_outs["deter"]), axis=1
        )
        stoch = jnp.concatenate(
            (sampled_data["stoch"][:, :1, ...], replay_outs["stoch"]), axis=1
        )
        sampled_data["deter"] = deter
        sampled_data["stoch"] = stoch

        sampled_data = tree_map(
            lambda val: einops.rearrange(val, "b t ... -> (b t) 1 ..."), sampled_data
        )
        rb_state.buffer = put2buffer(
            timestep_idxes, rb_state.buffer, sampled_data, env_idxes=env_idxes
        )  # because of the shape.
        learning_state = loss_and_info[0]
        if debug:
            logger._write(loss_and_info[2], env_fn.num_envs * idx)
    return {
        "key": key,
        "agent_modules": agent_modules,
        "policy_state": policy_state,
        "imag_state": imag_state,
        "opt_state": opt_state,
        "env_state": env_state,
        "rb_state": rb_state,
    }
