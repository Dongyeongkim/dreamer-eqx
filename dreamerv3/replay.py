import jax
import chex
import einops
import equinox as eqx
from jax import random
import jax.numpy as jnp
from jax.core import ShapedArray
from jax._src.lib import xla_client as xc
from jax.sharding import SingleDeviceSharding
from jax.tree_util import tree_map, tree_structure, tree_leaves, tree_unflatten

from typing import Dict


@chex.dataclass
class ReplayBuffer:
    left: list
    cache: dict
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
        cache={},
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


@eqx.filter_jit
def chunking(
    left, num_env_size, chunk_size, input_pytreedef, deskeydim, chunk_size_dict
):
    neg_splitpoint = (num_env_size * len(left)) % chunk_size
    splitpoint = len(left) - (neg_splitpoint // num_env_size)

    restored = [tree_unflatten(input_pytreedef, data) for data in left[:splitpoint]]
    if len(restored) == 0:
        return None

    prechunk = tree_stack(restored)
    prechunk = tree_map(transform2ds, prechunk, deskeydim)
    prechunk = tree_map(
        lambda val, chunk_size: einops.rearrange(
            val, "(t b) ... -> b t ...", t=chunk_size
        ),
        prechunk,
        chunk_size_dict,
    )
    return splitpoint, prechunk


def vectorize_cond_dict(pred, true_fun, false_fun, *operand_dicts):
    def apply_cond(p, *x):
        return jax.lax.cond(p, true_fun, false_fun, *x)

    return jax.tree_util.tree_map(
        lambda pred, *x: jax.vmap(apply_cond)(pred, *x), pred, *operand_dicts
    )


def get_from_buffer(idxes, buffer):
    buffer = tree_map(
        lambda idxes, val: putarray(
            jnp.take(val, idxes, axis=0), device=jax.devices()[0]
        ),
        idxes,
        buffer,
    )
    return buffer


def get_from_cachedbuffer(prechunks, idxes, bufferlen, deskeydim):
    preds = tree_map(
        lambda idx, blen: jnp.greater_equal(idx, blen),
        idxes,
        {k: bufferlen for k in deskeydim.keys()},
    )
    mod_idxes = tree_map(
        lambda idx, blen, prechunk: jnp.int32(jnp.greater_equal(idx, blen))
        * (idx - blen)
        + jnp.int32(jnp.less(idx, blen)) * (prechunk.shape[0]),
        idxes,
        {k: bufferlen for k in deskeydim.keys()},
        prechunks,
    )
    prechunks = tree_map(
        lambda idxes, val: jnp.take(val, idxes, axis=0),
        mod_idxes,
        prechunks,
    )
    return preds, prechunks


def optimised_sampling(buffer, bufferlen, prechunks, idxes, deskeydim):
    if bufferlen:
        buffer = get_from_buffer(idxes, buffer)
    else:
        _, sampled = get_from_cachedbuffer(prechunks, idxes, bufferlen, deskeydim)
        return sampled
    preds, prechunks = get_from_cachedbuffer(prechunks, idxes, bufferlen, deskeydim)
    sampled = vectorize_cond_dict(
        preds,
        lambda buffer, prechunk: prechunk,
        lambda buffer, prechunk: buffer,
        buffer,
        prechunks,
    )
    return sampled


def defragmenter(key, buffer_state, defrag_ratio, replay_ratio):
    key, partial_key = random.split(key, num=2)
    splitpoint, prechunks = chunking(
        buffer_state.left,
        buffer_state.num_env_size,
        buffer_state.chunk_size,
        buffer_state.input_pytreedef,
        buffer_state.deskeydim,
        buffer_state.chunk_size_dict,
    )
    buffer_state.left = buffer_state.left[splitpoint:]
    if prechunks is None:
        return key, buffer_state

    bufferlen = 0
    if len(buffer_state.buffer) == 0:
        idxes = random.randint(
            partial_key,
            shape=(defrag_ratio // replay_ratio,),
            minval=0,
            maxval=len(list(prechunks.values())[0]),
        )
    else:
        idxes = random.randint(
            partial_key,
            shape=(defrag_ratio // replay_ratio,),
            minval=0,
            maxval=len(list(buffer_state.buffer.values())[0])
            + len(list(prechunks.values())[0]),
        )
        bufferlen = len(list(buffer_state.buffer.values())[0])

    idxes_dict = {k: idxes for k in buffer_state.deskeydim.keys()}
    buffer_state.cache = optimised_sampling(
        buffer_state.buffer,
        bufferlen,
        prechunks,
        idxes_dict,
        buffer_state.deskeydim,
    )
    prechunks_cpu = tree_map(
        lambda val: putarray(val, jax.devices("cpu")[0]), prechunks
    )
    if bufferlen:
        buffer_state.buffer = tree_concat([buffer_state.buffer, prechunks_cpu])
    else:
        buffer_state.buffer = prechunks_cpu

    return key, buffer_state


def sampler(
    idx,
    cache,
    deskeydim,
    batch_size_dict,
    batch_length_dict,
    defrag_ratio,
    replay_ratio,
):
    idx %= defrag_ratio // replay_ratio
    idxes = {k: idx for k in deskeydim.keys()}
    sampled = tree_map(lambda idx, val: val[idx].squeeze(), idxes, cache)
    batched = tree_map(
        transform2batch,
        sampled,
        deskeydim,
        batch_size_dict,
        batch_length_dict,
    )
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
        "(b t) ... -> b t ...",
        b=batch_size,
        t=batch_length,
    )
