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
    buffer_ptr: int
    online_ptr: int
    fragment_ptr: int
    bufferlen_per_env: int
    num_env: int
    batch_size: int
    batch_length: int


def generate_replaybuffer(
    buffer_size, desired_key_dtype_dim, batch_size, batch_length, num_env=1
):
    assert (
        num_env > 0
    ), "number of the environments should be greater or equal than 1(replay buffer)"

    blen_per_env = ceil(buffer_size / num_env)

    fragment_desired_key_and_dim = {
        k: (
            (
                batch_length,
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
                num_env,
                blen_per_env,
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
        buffer_ptr=0,
        online_ptr=0,
        fragment_ptr=0,
        bufferlen_per_env=blen_per_env,
        num_env=num_env,
        batch_size=batch_size,
        batch_length=batch_length,
    )


def calcbufferidxes(buffer_ptr, buffer_size, batch_length):
    if buffer_ptr + batch_length >= buffer_size:
        is_full = True
        next_buffer_ptr = buffer_ptr + batch_length - buffer_size
        idxes = jnp.concatenate(
            (
                jnp.arange(start=buffer_ptr, stop=buffer_size),
                jnp.arange(start=0, stop=next_buffer_ptr),
            )
        )

        return is_full, next_buffer_ptr, idxes

    else:
        is_full = False
        next_buffer_ptr = buffer_ptr + batch_length
        idxes = jnp.arange(start=buffer_ptr, stop=buffer_ptr + batch_length)

        return is_full, next_buffer_ptr, idxes


def calconlineidxes(key, buffer_ptr, online_ptr, bufferlen, batch_size, batch_length):
    if buffer_ptr > online_ptr:
        num_chunks = (buffer_ptr - online_ptr) // batch_length
        if num_chunks >= batch_size:
            idxes = jnp.arange(start=online_ptr, stop=online_ptr + batch_size)
            online_ptr += batch_size
        else:
            online_idxes = jnp.arange(start=online_ptr, stop=online_ptr + num_chunks)
            sample_idxes = jax.random.randint(
                key,
                shape=(batch_size - num_chunks,),
                minval=0,
                maxval=bufferlen - batch_length,
            )
            idxes = jnp.concatenate([online_idxes, sample_idxes])
            online_ptr += num_chunks
        return idxes, online_ptr
    else:
        idxes = jax.random.randint(
            key, shape=(batch_size,), minval=0, maxval=bufferlen - batch_length
        )
        return idxes, online_ptr


def calcfragmentidxes(fragment_id, fragment_size):
    assert fragment_id > -1, "the ptr must be in positive integer space"
    if fragment_id == fragment_size:
        return 0
    else:
        return fragment_id + 1


def sampler(
    key,
    buffer,
    num_envs,
    bufferptr,
    onlineptr,
    bufferlen,
    batch_size,
    batch_length,
    env_idx=None,
):
    env_idx_key, timestep_idx_key = jax.random.split(key, num=2)
    partial_env_idxes = (
        jax.random.randint(env_idx_key, shape=(batch_size,), minval=0, maxval=num_envs)
        if env_idx is None
        else env_idx
    )  # have a defect; Not sure on multi-env setting at this point.
    env_idxes = jnp.repeat(partial_env_idxes, repeats=batch_length)
    timestep_idxes, onlineptr = calconlineidxes(
        timestep_idx_key, bufferptr, onlineptr, bufferlen, batch_size, batch_length
    )
    idxes = (
        jnp.linspace(timestep_idxes, timestep_idxes + batch_length - 1, batch_length)
        .astype("int32")
        .swapaxes(0, 1)
        .flatten()
    )
    with jax.default_device(jax.devices("cpu")[0]):
        raw_sampled = tree_map(lambda val: val[env_idxes, idxes, ...].copy(), buffer)
    sampled = tree_map(
        lambda val: einops.rearrange(val, "(b t) ... -> b t ...", b=batch_size),
        raw_sampled,
    )
    sampled = jax.device_put(sampled, device=jax.devices()[0])
    return env_idxes, idxes, onlineptr, sampled


@partial(jax.jit, donate_argnums=1)
def put2fragmentcache(idx, fragmentcache, timestep):
    fragmentcache = tree_map(
        lambda cache, val: cache.at[idx].set(val), fragmentcache, timestep
    )
    return fragmentcache


# Donating the memory is not recommended if the input shape and output shape is different, but it is necessary because of the performance issue.
# it is allowed on CPU/GPUs(NOT on TPUs because of not-reconfigurable memory architecture; https://github.com/google/jax/issues/11036)
@partial(jax.jit, donate_argnums=(1, 2), device=jax.devices("cpu")[0])
def put2buffer(indices, buffer, fragmentcache, env_idxes=None):
    if env_idxes is None:
        buffer = tree_map(
            lambda buffer, val: buffer.at[:, indices].set(
                einops.rearrange(val, "t b ... -> b t ...")
            ),
            buffer,
            fragmentcache,
        )
    else:
        buffer = tree_map(
            lambda buffer, val: buffer.at[env_idxes, indices].set(
                einops.rearrange(val, "t b ... -> b t ...").squeeze()
            ),
            buffer,
            fragmentcache,
        )
    return buffer
