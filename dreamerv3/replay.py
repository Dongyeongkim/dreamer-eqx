import jax
import chex
import einops
from math import ceil
import jax.numpy as jnp
from functools import partial
from jax.tree import map as tree_map


@chex.dataclass
class ReplayBuffer:
    fragment: dict
    buffer: dict
    is_full: bool
    chunk_ptr: int
    num_chunks: int
    fragment_ptr: int
    num_fragment: int
    num_env: int
    batch_size: int


def generate_replaybuffer(
    buffer_size, desired_key_dtype_dim, batch_size, batch_length, num_env=1
):
    assert (
        num_env > 0
    ), "number of the environments should be greater or equal than 1(replay buffer)"

    n_chunks = ceil(buffer_size / (batch_size * batch_length))

    fragment_desired_key_and_dim = {
        k: (
            (
                batch_size * batch_length,
                num_env,
            )
            + v[1],
            v[0],
        )
        for k, v in desired_key_dtype_dim.items()
    }
    fragment_initial = {
        k: jnp.zeros(v[0], dtype=v[1]) for k, v in fragment_desired_key_and_dim.items()
    }

    buffer_desired_key_and_dim = {
        k: (
            (
                n_chunks,
                batch_size * batch_length,
            )
            + v[1],
            v[0],
        )
        for k, v in desired_key_dtype_dim.items()
    }
    with jax.default_device(jax.devices("cpu")[0]):
        buffer_initial = {
            k: jnp.zeros(v[0], dtype=v[1])
            for k, v in buffer_desired_key_and_dim.items()
        }

    return ReplayBuffer(
        fragment=fragment_initial,
        buffer=buffer_initial,
        is_full=False,
        chunk_ptr=0,
        num_chunks=n_chunks,
        fragment_ptr=0,
        num_fragment=batch_size * batch_length,
        num_env=num_env,
        batch_size=batch_size,
    )


def calcbufferidxes(chunk_id, buffer_size, num_env):
    if chunk_id + num_env >= buffer_size:
        is_full = True
        next_chunk_ptr = chunk_id + num_env - buffer_size
        idxes = jnp.concatenate(
            (
                jnp.arange(start=chunk_id, stop=buffer_size),
                jnp.arange(start=0, stop=chunk_id + num_env - buffer_size),
            )
        )
        
        return is_full, next_chunk_ptr, idxes
    
    else:
        is_full = False
        next_chunk_ptr = chunk_id + num_env
        idxes = jnp.arange(start=chunk_id, stop=chunk_id + num_env)

        return is_full, next_chunk_ptr, idxes


def calcfragmentidxes(fragment_id, fragment_size):
    assert fragment_id > -1, "the ptr must be in positive integer space"
    if fragment_id == fragment_size:
        return 0
    else:
        return fragment_id + 1


def sampler(key, bufferlen, buffer, batch_size, idx=None):
    idx = jax.random.randint(key, shape=(1,), minval=0, maxval=bufferlen) if idx is None else idx
    raw_sampled = tree_map(lambda val: jnp.take(val, idx, axis=0).squeeze(), buffer)
    sampled = tree_map(
        lambda val: einops.rearrange(val, "(b t) ... -> b t ...", b=batch_size),
        raw_sampled,
    )
    sampled = jax.device_put(sampled, device=jax.devices()[0])
    return idx, sampled


@partial(jax.jit, donate_argnums=1)
def put2fragmentcache(idx, fragmentcache, timestep):
    fragmentcache = tree_map(
        lambda cache, val: cache.at[idx].set(val), fragmentcache, timestep
    )
    return fragmentcache


# Donating the memory is not recommended if the input shape and output shape is different, but it is necessary because of the performance issue.
# it is allowed on CPU/GPUs(NOT on TPUs because of not-reconfigurable memory architecture; https://github.com/google/jax/issues/11036)
@partial(jax.jit, donate_argnums=(1, 2), device=jax.devices("cpu")[0])
def put2buffer(indices, buffer, fragmentcache):
    buffer = tree_map(
        lambda buffer, val: buffer.at[indices].set(
            einops.rearrange(val, "t b ... -> b t ...")
        ),
        buffer,
        fragmentcache,
    )
    return buffer
