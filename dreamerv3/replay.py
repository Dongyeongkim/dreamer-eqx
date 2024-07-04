import jax
import chex
import einops
import equinox as eqx
from jax import random
import jax.numpy as jnp
from jax.tree_util import tree_map, tree_structure, tree_leaves, tree_unflatten

from typing import Dict
from functools import partial


@chex.dataclass
class ReplayBuffer:
    left: list
    buffer: dict
    chunk_id: int
    deskeydim: dict
    chunk_size: int
    batch_size: int
    batch_length: int
    num_env_size: int
    buffer_size: int
    input_pytreedef: dict
    chunk_size_dict: dict
    batch_size_dict: dict
    batch_length_dict: dict

def generate_replaybuffer(
    buffer_size, desired_key_dim, batch_size, batch_length, num_env=1
):
    assert (
        num_env > 0
    ), "number of the environments should be greater or equal than 1(replay buffer)"
    desired_key_and_dim = {k: len((num_env,) + v) for k, v in desired_key_dim.items()}
    return ReplayBuffer(
        left=[],
        buffer={},
        chunk_id=0,
        deskeydim=desired_key_and_dim,
        chunk_size=batch_size * batch_length,
        batch_size=batch_size,
        batch_length=batch_length,
        num_env_size=num_env,
        buffer_size=buffer_size,
        input_pytreedef=tree_structure(desired_key_and_dim),
        chunk_size_dict={
            k: batch_size * batch_length for k in desired_key_and_dim.keys()
        },
        batch_size_dict={k: batch_size for k in desired_key_and_dim.keys()},
        batch_length_dict={k: batch_length for k in desired_key_and_dim.keys()},
    )


# No JIT; JItting will make the performance poor. The reason is it needs to optimise its shape of left while dynamically changed it


def pushstep(buffer_state, data: Dict[str, jnp.array]):
    vals = tree_leaves(data)
    buffer_state.left.append(vals)
    return buffer_state


def defragmenter(buffer_state):
    neg_splitpoint = (
        buffer_state.num_env_size * len(buffer_state.left)
    ) % buffer_state.chunk_size
    splitpoint = len(buffer_state.left) - (neg_splitpoint // buffer_state.num_env_size)

    restored = [
        tree_unflatten(buffer_state.input_pytreedef, data)
        for data in buffer_state.left[:splitpoint]
    ]
    if len(restored) == 0:
        return buffer_state

    buffer_state.left = buffer_state.left[splitpoint:]

    prechunk = tree_stack(restored)
    prechunk = tree_map(transform2ds, prechunk, buffer_state.deskeydim)
    prechunk = tree_map(
        lambda val, chunk_size: einops.rearrange(
            val, "(t c) ... -> t c ...", c=chunk_size
        ),
        prechunk,
        buffer_state.chunk_size_dict,
    )
    if len(buffer_state.buffer.keys()) == 0:
        buffer_state.buffer = prechunk
        return buffer_state

    else:
        buffer_state.buffer = tree_concat([buffer_state.buffer, prechunk])

    return buffer_state


def sampler(key, buffer_state, device=None):
    idx = random.randint(
        key, shape=(), minval=0, maxval=len(list(buffer_state.buffer.values())[0])
    )
    idxes = {k: idx for k in buffer_state.deskeydim.keys()}
    sampled = tree_map(lambda idx, val: val[idx].squeeze(), idxes, buffer_state.buffer)
    batched = tree_map(
        transform2batch,
        sampled,
        buffer_state.deskeydim,
        buffer_state.batch_size_dict,
        buffer_state.batch_length_dict,
    )
    batched = putarray(batched, jax.devices()[0] if device is None else device)
    return batched


def tree_stack(trees):
    return tree_map(lambda *v: jnp.stack(v), *trees)


def tree_concat(trees):
    return tree_map(lambda *v: jnp.concatenate(v), *trees)


def putarray(data, device):
    return jax.device_put(data, device)


def transform2ds(data: jnp.array, expected_dim: int):
    # IMPORTANT: EXPECTED SHAPE -> (T, B, ...)
    assert (
        len(data.shape) < expected_dim + 2
    ), " dimension cannot be bigger than expected shape + batch dimension"
    assert (
        len(data.shape) > expected_dim - 1
    ), " dimension cannot be smaller than expected shape"
    if len(data.shape) == expected_dim:
        return data
    elif len(data.shape) == expected_dim + 1:
        return einops.rearrange(data, "t b ... -> (t b) ...")
    else:
        raise NotImplementedError("Something is wrong")


def transform2batch(
    data: jnp.array, expected_dim: int, batch_size: int, batch_length: int
):
    assert len(data.shape) == expected_dim, "dimension does not fit with expected dim"
    return einops.rearrange(
        data,
        "(t b) ... -> b t ...",
        b=batch_size,
        t=batch_length,
    )

