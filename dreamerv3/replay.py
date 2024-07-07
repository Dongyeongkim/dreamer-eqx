import jax
import chex
import einops
import jax.numpy as jnp
from functools import partial
from jax.tree import map as tree_map

@chex.dataclass
class ReplayBuffer:
    cache: dict
    buffer: dict
    is_full: bool
    chunk_id: int
    


def generate_replaybuffer():
    pass


def calcidxes(chunk_id, buffer_size, num_env):
    if chunk_id + num_env >= buffer_size:
        return jnp.concatenate(
            (
                jnp.arange(start=chunk_id, stop=buffer_size),
                jnp.arange(start=0, stop=chunk_id + num_env - buffer_size),
            )
        )
    else:
        return jnp.arange(start=chunk_id, stop=(chunk_id + num_env))


def sampler(key, bufferlen, buffer, batch_size):
    idx = jax.random.randint(key, shape=(1,), minval=0, maxval=bufferlen)
    raw_sampled = tree_map(lambda val: jnp.take(val, idx, axis=0).squeeze(), buffer)
    sampled = tree_map(
        lambda val: einops.rearrange(val, "(b t) ... -> b t ...", b=batch_size), raw_sampled
    )
    sampled = jax.device_put(sampled, device=jax.devices()[0])
    return idx, sampled


@partial(jax.jit, donate_argnums=1)
def put2fragmentcache(idx, fragmentcache, timestep):
    fragmentcache = tree_map(
        lambda cache, val: cache.at[idx].set(val), fragmentcache, timestep
    )
    return fragmentcache


# Donating the memory is not recommended if the input shape and output shape is different, but necessary because of the performance issue.
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