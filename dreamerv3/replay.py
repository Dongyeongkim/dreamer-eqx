import jax
import einops
import equinox as eqx
from jax import random
import jax.numpy as jnp
from jax.tree_util import tree_map, tree_structure, tree_leaves, tree_unflatten

from typing import Dict

import chex


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
    if num_env == 1:
        desired_key_and_dim = {k: len(v) for k, v in desired_key_dim.items()}
    else:
        desired_key_and_dim = {
            k: len((num_env,) + v) for k, v in desired_key_dim.items()
        }
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
    buffer_state.left = [*buffer_state.left, vals]
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
            val, "(t c) ... -> t c ...", c=chunk_size
        ),
        prechunk,
        chunk_size_dict,
    )
    return splitpoint, prechunk


@eqx.filter_jit
def optimised_sampling(
    buffer, bufferlen, prechunk, deskeydim, defrag_ratio, replay_ratio
):
    len_idxes = defrag_ratio // replay_ratio
    idxes = jnp.arange(len_idxes)
    indicator = jnp.where(idxes == bufferlen, True, False)
    if indicator.all() == True:
        cache = tree_map(lambda val: jnp.repeat(val, len_idxes, axis=0), prechunk)
        return cache
    elif indicator.any() == False:
        idxes = {k: idxes for k in deskeydim.keys()}
        sampled = tree_map(
            lambda idxes, val: jnp.take_along_axis(val, idxes, axis=0), idxes, buffer
        )
        return putarray(sampled)
    else:
        trees = []
        for idx, ind in enumerate(indicator):
            if ind:
                trees.append(prechunk)
            else:
                idxes = {k: jnp.array(idx) for k in deskeydim.keys()}
                cpu_sampled = tree_map(
                    lambda idx, val: putarray(jnp.take_along_axis(val, idx, axis=0)),
                    idxes,
                    buffer,
                )
                trees.append(cpu_sampled)
        sampled = tree_stack(trees)
        return sampled


def defragmenter(key, buffer_state, defrag_ratio, replay_ratio):
    key, partial_key = random.split(key, num=2)
    splitpoint, prechunk = chunking(
        buffer_state.left,
        buffer_state.num_env_size,
        buffer_state.chunk_size,
        buffer_state.input_pytreedef,
        buffer_state.deskeydim,
        buffer_state.chunk_size_dict,
    )
    buffer_state.left = buffer_state.left[splitpoint:]
    if prechunk is None:
        return key, buffer_state
    if len(buffer_state.buffer.keys()) == 0:
        buffer_state.cache = tree_map(
            lambda val: jnp.repeat(val, defrag_ratio // replay_ratio, axis=0), prechunk
        )
        prechunk_cpu = putarray(prechunk, jax.devices("cpu")[0])
        buffer_state.buffer = prechunk_cpu

    else:
        buffer_state.cache = optimised_sampling(
            buffer_state.buffer,
            prechunk,
            buffer_state.deskeydim,
            defrag_ratio,
            replay_ratio,
        )
        prechunk = putarray(prechunk, jax.devices("cpu")[0])
        buffer_state.buffer = tree_concat([buffer_state.buffer, prechunk])

    return key, buffer_state


@eqx.filter_jit
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


def optimisedgetchunk(data, chunk_length: int):
    chunks = jnp.array_split(data, jnp.arange(chunk_length, len(data), chunk_length))
    return chunks


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


@eqx.filter_jit
def testfunc(key, buffer_ds):
    for i in range(2000):
        buffer_ds = pushstep(
            buffer_ds, {"obs": jnp.zeros((16, 64, 64, 3)), "action": jnp.zeros((16, 6))}
        )
        key, sampling_key = random.split(key)
        data = sampler(sampling_key, buffer_ds)
        if i % 65 == 0 and i != 0:
            buffer_ds = defragmenter(buffer_ds)
    return buffer_ds


if __name__ == "__main__":
    buffer_ds = generate_replaybuffer(
        buffer_size=1_000_000,
        desired_key_dim={"obs": (64, 64, 3), "action": (6,)},
        batch_size=16,
        batch_length=65,
        num_env=16,
    )
    for i in range(65):
        buffer_ds = pushstep(
            buffer_ds, {"obs": jnp.zeros((16, 64, 64, 3)), "action": jnp.zeros((16, 6))}
        )
    buffer_ds = defragmenter(buffer_ds)
    import time

    a = time.time()
    buffer_ds = testfunc(jax.random.key(0), buffer_ds)
    b = time.time()
    print(16 * 1000 / (b - a))
    print(buffer_ds.buffer["action"].shape)
