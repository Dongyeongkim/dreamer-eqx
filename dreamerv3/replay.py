import jax
import einops
import equinox as eqx
import jax.numpy as jnp
from jax.tree_util import tree_map, tree_structure, tree_leaves, tree_unflatten

from typing import Dict

import chex

@chex.dataclass
class ReplayBuffer:
    left: list
    buffer: dict
    chunk_id: int
    deskeydim: dict
    buffer_size: int
    input_pytreedef: chex.PyTreeDef
    batch_size_dict: dict
    chunk_size_dict: dict


def generate_replaybuffer(buffer_size, desired_key_dim, batch_size, chunk_size, num_env=1):
    assert num_env > 0, "number of the environments should be greater or equal than 1(replay buffer)"
    if num_env == 1:
        desired_key_and_dim = {k: jnp.zeros(shape=v) for k,v in desired_key_dim.items()}
    else:
        desired_key_and_dim = {k: jnp.zeros(shape=((num_env,)+v)) for k,v in desired_key_dim.items()}
    return ReplayBuffer(
        left=[],
        buffer={},
        chunk_id=0,
        deskeydim={k: len(v.shape) for k,v in desired_key_and_dim.items()},
        buffer_size=buffer_size,
        input_pytreedef=tree_structure(desired_key_and_dim),
        batch_size_dict={k: batch_size for k in desired_key_and_dim.keys()},
        chunk_size_dict={k: chunk_size for k in desired_key_and_dim.keys()},
    )

#@eqx.filter_jit
def pushstep(buffer_state, data: Dict[str, jnp.array]):
    data = tree_map(transform2ds, data, buffer_state.deskeydim)
    vals = tree_leaves(data)
    buffer_state.replace(
        left=buffer_state.left.append(vals)
    )
    return buffer_state

@eqx.filter_jit
def defragmenter(buffer_state):
    restored = [tree_unflatten(buffer_state.input_pytreedef, data) for data in buffer_state.left]
    breakpoint()
    prechunk = tree_map(getleft, buffer_state.left)
    chunks_and_infos = tree_map(optimisedgetchunk, prechunk)
    chunk_cond = {k: v[0] for k, v in chunks_and_infos.items()}
    leftchunk_cond = [v[1] for v in chunks_and_infos.values()]
    numchunks = [v[2] for v in chunks_and_infos.values()]
    chunks = {k: v[3] for k, v in chunks_and_infos.items()}
    if True in chunk_cond.values():
        buffer_state.left = {k: optimisedgetleft(v) for k, v in chunks.items()}
        return buffer_state

    else:
        if leftchunk_cond[0]:
            for i in range(numchunks[0]):
                idx = buffer_state.chunk_id % (
                    buffer_state.buffer_size // buffer_state.chunk_size
                )
                buffer_state.buffer.update({idx: {k: v[i] for k, v in chunks.items()}})
                buffer_state.chunk_id += 1

            buffer_state.left = {k: optimisedgetleft(v) for k, v in chunks.items()}
            return buffer_state

        else:
            for i in range(numchunks[0]):
                idx = buffer_state.chunk_id % (
                    buffer_state.buffer_size // buffer_state.chunk_size
                )
                buffer_state.buffer.update({idx: {k: v[i] for k, v in chunks.items()}})
                buffer_state.chunk_id += 1

            buffer_state.left = {k: jnp.array([]) for k in chunks.keys()}
            return buffer_state


@eqx.filter_jit
def sampler(key, buffer_state, device=None):
    if buffer_state.chunk_id > buffer_state.buffer_size // buffer_state.chunk_size:
        idx = int(
            jax.random.choice(
                key, jnp.arange(buffer_state.buffer_size // buffer_state.chunk_size)
            )
        )
    else:
        idx = int(jax.random.choice(key, jnp.arange(buffer_state.chunk_id)))
    data = tree_map(
        transform2batch,
        buffer_state.buffer[idx],
        buffer_state.deskeydim,
        buffer_state.batch_size,
        buffer_state.chunk_size,
    )
    data = tree_map(
        poparray,
        data,
        {
            k: jax.devices()[0] if device is None else device
            for k in buffer_state.deskeydim.keys()
        },
    )
    return buffer_state, data


def pusharray(data, leftover):
    if len(leftover) == 0:
        return jax.device_put(data, device=jax.devices("cpu")[0])
    else:
        return jax.lax.concatenate(
            [leftover, jax.device_put(data, device=jax.devices("cpu")[0])], 0
        )

def restoretree(tree, data):
    return tree_unflatten(tree, data)

def poparray(data, device):
    return jax.device_put(data, device)


def optimisedgetchunk(data, chunk_size: int):
    return (
        ((len(data) % chunk_size) == len(data)),
        (len(data) % chunk_size),
        (len(data) // chunk_size),
        jnp.array_split(data, jnp.arange(chunk_size, len(data), chunk_size)),
    )


def optimisedgetleft(data):
    return data[-1]


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
    data: jnp.array, expected_dim: int, batch_size: int, chunk_size: int
):
    assert len(data.shape) == expected_dim, "dimension does not fit with expected dim"
    return einops.rearrange(
        data,
        "(t b) ... -> b t ...",
        b=batch_size,
        t=chunk_size // batch_size,
    )


if __name__ == "__main__":
    buffer_ds = generate_replaybuffer(buffer_size=1_000_000, desired_key_dim={"obs": (64, 64, 3), "action": (6,)}, batch_size=16, chunk_size=64, num_env=16)
    for i in range(2000):
        buffer_ds = pushstep(buffer_ds, {"obs": jnp.zeros((16, 64, 64, 3)), "action": jnp.zeros((16, 6))})
    
    defragmenter(buffer_ds)
    