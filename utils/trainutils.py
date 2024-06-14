import jax
from jax import random
import jax.numpy as jnp
from dreamerv3.replay import defragmenter, pushstep, sampler


# rollout_fn
#   - interaction_fn: interacting with jax environment
#   - train_wm_fn: training wm with function

def rollout_fn(
    key,
    num_steps,
    defrag_ratio,
    replay_ratio,
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
    def step_fn(carry, idx):
        if idx % defrag_ratio == 0 and idx != 0:
            rb_state = defragmenter(rb_state)

        if idx % replay_ratio == 0 and idx != 0:
            carry = train_wm_fn(agent_fn, env_fn, opt_fn, **carry)
            return carry, _
        else:
            carry = interaction_fn(agent_fn, env_fn, opt_fn, **carry)
            return carry, _

    init_carry = {
        "key": key,
        "agent_modules": agent_modules,
        "agent_state": agent_state,
        "opt_state": opt_state,
        "env_state": env_state,
        "rb_state": rb_state,
    }

    carry, _ = jax.lax.scan(step_fn, init_carry, jnp.arange(num_steps), unroll=False)

    return carry


def interaction_fn(
    agent_fn,
    env_fn,
    opt_fn,
    interaction_key,
    agent_modules,
    agent_state,
    env_params,
    env_state,
    opt_state,
    rb_state,
):
    key, policy_key, env_key = random.split(interaction_key, num=3)
    env_state, timestep = env_fn.step(
        env_key, env_state, agent_state[0][1].argmax(axis=1), env_params
    )
    timestep["action"] = agent_state[0][1]
    agent_state, outs = agent_fn.policy(
        policy_key, agent_modules, agent_state, timestep
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


def train_wm_fn(
    agent_fn,
    env_fn,
    opt_fn,
    training_key,
    agent_modules,
    agent_state,
    env_params,
    env_state,
    opt_state,
    rb_state,
):
    key, rb_sampling_key, training_key = random.split(training_key, num=2)
    sampled_data = sampler(rb_sampling_key, rb_state)
    agent_modules, total_loss, loss_and_info, opt_state = agent_fn.train(
        training_key, agent_modules, agent_state, opt_fn, opt_state, sampled_data
    )
    return {
        "key": key,
        "agent_modules": agent_modules,
        "agent_state": agent_state,
        "opt_state": opt_state,
        "env_state": env_state,
        "rb_state": rb_state,
    }


# prefill_fn 
#   - interaction_fn: only interactions to fill replay buffer

def prefill_fn(
        key, 
        prefill_steps,

):
    pass