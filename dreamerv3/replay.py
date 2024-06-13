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
    batch_size: int
    chunk_size: int
    buffer_size: int
    input_pytreedef: chex.PyTreeDef
    batch_size_dict: dict
    chunk_size_dict: dict


def generate_replaybuffer(buffer_size, desired_key_dim, batch_size, chunk_size, num_env=1):
    assert num_env > 0, "number of the environments should be greater or equal than 1(replay buffer)"
    if num_env == 1:
        desired_key_and_dim = {k: len(v) for k,v in desired_key_dim.items()}
    else:
        desired_key_and_dim = {k: len((num_env,)+v) for k,v in desired_key_dim.items()}
    return ReplayBuffer(
        left=[],
        buffer={},
        chunk_id=0,
        deskeydim=desired_key_and_dim,
        batch_size=batch_size,
        chunk_size=chunk_size,
        buffer_size=buffer_size,
        input_pytreedef=tree_structure(desired_key_and_dim),
        batch_size_dict={k: batch_size for k in desired_key_and_dim.keys()},
        chunk_size_dict={k: chunk_size for k in desired_key_and_dim.keys()},
    )

# No JIT; JItting will make the performance poor. The reason is it needs to optimise its shape of left while dynamically changed it


def pushstep(buffer_state, data: Dict[str, jnp.array]):
    data = tree_map(transform2ds, data, buffer_state.deskeydim)
    vals = tree_leaves(data)
    buffer_state.left = [*buffer_state.left, vals]
    return buffer_state


def defragmenter(buffer_state):
    splitpoint = (len(buffer_state.left) // buffer_state.chunk_size) * buffer_state.chunk_size
    restored = [tree_unflatten(buffer_state.input_pytreedef, data) for data in buffer_state.left[:splitpoint]]
    numchunks = len(restored) // buffer_state.chunk_size
    if numchunks == 0:
        return buffer_state
    prechunk = tree_stack(restored)
    
    chunks = tree_map(optimisedgetchunk, prechunk, buffer_state.chunk_size_dict)
    for i in range(numchunks):
        idx = buffer_state.chunk_id % (buffer_state.buffer_size // buffer_state.chunk_size)
        buffer_state.buffer.update({idx: {k: v[i] for k, v in chunks.items()}})
        buffer_state.chunk_id += 1
    
    buffer_state.left = buffer_state.left[splitpoint:]
    return buffer_state


@eqx.filter_jit
def sampler(key, buffer_state, device=None):
    if buffer_state.chunk_id > buffer_state.buffer_size // buffer_state.chunk_size:
        idx = jax.random.choice(key, jnp.arange(buffer_state.buffer_size // buffer_state.chunk_size))
    else:
        idx = jax.random.choice(key, jnp.arange(buffer_state.chunk_id))
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


def tree_stack(trees):
    return jax.device_put(tree_map(lambda *v: jnp.stack(v), *trees), device=jax.devices("cpu")[0])

def poparray(data, device):
    return jax.device_put(data, device)


def optimisedgetchunk(data, chunk_size: int):
    return jnp.array_split(data, jnp.arange(chunk_size, len(data), chunk_size))


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


@eqx.filter_jit
def testfunc(buffer_ds):
    for i in range(1000):
        for _ in range(2):
            buffer_ds = pushstep(buffer_ds, {"obs": jnp.zeros((16, 64, 64, 3)), "action": jnp.zeros((16, 6))})
        if i % 520:
            buffer_ds = defragmenter(buffer_ds)
    return buffer_ds
    

if __name__ == "__main__":
    buffer_ds = generate_replaybuffer(buffer_size=1_000_000, desired_key_dim={"obs": (64, 64, 3), "action": (6,)}, batch_size=16, chunk_size=1040, num_env=16)
    import time
    a = time.time()
    buffer_ds = testfunc(buffer_ds)
    b = time.time()
    print(b-a)
    buffer_ds, data = sampler(jax.random.key(0), buffer_ds)