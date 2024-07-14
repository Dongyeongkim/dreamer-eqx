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
    fragment_size: int
    bufferlen_per_env: int
    num_env: int
    batch_size: int


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
        fragment_size=batch_size * batch_length,
        bufferlen_per_env=blen_per_env,
        num_env=num_env,
        batch_size=batch_size,
    )


def calcbufferidxes(buffer_ptr, buffer_size, chunk_size):
    if buffer_ptr + chunk_size >= buffer_size:
        is_full = True
        next_buffer_ptr = buffer_ptr + chunk_size - buffer_size
        idxes = jnp.concatenate(
            (
                jnp.arange(start=buffer_ptr, stop=buffer_size),
                jnp.arange(start=0, stop=next_buffer_ptr),
            )
        )

        return is_full, next_buffer_ptr, idxes

    else:
        is_full = False
        next_buffer_ptr = buffer_ptr + chunk_size
        idxes = jnp.arange(start=buffer_ptr, stop=buffer_ptr + chunk_size)

        return is_full, next_buffer_ptr, idxes


def calcfragmentidxes(fragment_id, fragment_size):
    assert fragment_id > -1, "the ptr must be in positive integer space"
    if fragment_id == fragment_size:
        return 0
    else:
        return fragment_id + 1


def sampler(
    key,
    bufferlen,
    buffer,
    batch_size,
    chunk_size,
    num_envs,
    env_idx=None,
    timestep_idx=None,
):
    env_idx_key, timestep_idx_key = jax.random.split(key, num=2)
    env_idx = (
        jax.random.randint(env_idx_key, shape=(), minval=0, maxval=num_envs)
        if env_idx is None
        else env_idx
    )
    timestep_idx = (
        jax.random.randint(
            timestep_idx_key, shape=(), minval=0, maxval=bufferlen - chunk_size
        )
        if timestep_idx is None
        else timestep_idx
    )
    idxes = jnp.arange(start=timestep_idx, stop=timestep_idx + chunk_size)
    raw_sampled = tree_map(lambda val: jnp.take(val, idxes, axis=1)[env_idx], buffer)
    sampled = tree_map(
        lambda val: einops.rearrange(val, "(b t) ... -> b t ...", b=batch_size),
        raw_sampled,
    )
    sampled = jax.device_put(sampled, device=jax.devices()[0])
    return env_idx, idxes, sampled


@partial(jax.jit, donate_argnums=1)
def put2fragmentcache(idx, fragmentcache, timestep):
    fragmentcache = tree_map(
        lambda cache, val: cache.at[idx].set(val), fragmentcache, timestep
    )
    return fragmentcache


# Donating the memory is not recommended if the input shape and output shape is different, but it is necessary because of the performance issue.
# it is allowed on CPU/GPUs(NOT on TPUs because of not-reconfigurable memory architecture; https://github.com/google/jax/issues/11036)
@partial(jax.jit, donate_argnums=(1, 2), device=jax.devices("cpu")[0])
def put2buffer(indices, buffer, fragmentcache, env_idx=None):
    if env_idx is None:
        buffer = tree_map(
            lambda buffer, val: buffer.at[:, indices].set(
                einops.rearrange(val, "t b ... -> b t ...")
            ),
            buffer,
            fragmentcache,
        )
    else:
        buffer = tree_map(
            lambda buffer, val: buffer.at[jnp.array([env_idx]), indices].set(
                einops.rearrange(val, "t b ... -> b t ...").squeeze()
            ),
            buffer,
            fragmentcache,
        )
    return buffer
