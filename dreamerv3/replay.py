import jax
import einops
import jax.numpy as jnp
from jax.tree_util import tree_map
from typing import Dict
from functools import partial


class List(list):
    pass


@partial(jax.jit, static_argnums=(1,))
def leftoverarray(data, chunk_size):
    slice_size = len(data) // chunk_size * chunk_size
    return jax.lax.dynamic_slice(data, (slice_size, *data.shape[1:]), (len(data) - slice_size, *data.shape[1:]))


@partial(jax.jit, static_argnums=(1, 2, 3,))
def transform2batch(data: jnp.array, expected_dim: int, chunk_size: int, batch_size: int):
    assert len(data.shape) == expected_dim, "dimension does not fit with expected dim"
    return einops.rearrange(data, '(t b) ... -> b t ...', b=batch_size,
                            t=chunk_size // batch_size)


@partial(jax.jit, static_argnums=(1,))
def getarrays(data, chunk_size):
    slice_size = len(data) // chunk_size * chunk_size
    return data[:slice_size].reshape(-1, chunk_size, *data.shape[1:])


# @partial(jax.jit, static_argnums=(1,))
def transform2ds(data: jnp.array, expected_dim: int):
    # IMPORTANT: EXPECTED SHAPE -> (T, B, ...)
    # assert len(data.shape) < expected_dim + 2, " dimension cannot be bigger than expected shape + batch dimension"
    # assert len(data.shape) > expected_dim - 1, " dimension cannot be smaller than expected shape"
    if len(data.shape) == expected_dim:
        return data
    elif len(data.shape) == expected_dim + 1:
        return einops.rearrange(data, 't b ... -> (t b) ...')
    else:
        raise NotImplementedError("Something is wrong")


class ReplayBuffer:
    def __init__(self, buffer_size, key_and_desired_dim, batch_size, chunk_size=1024):
        assert chunk_size % batch_size == 0
        self.buffer_size = buffer_size
        self.deskeydim = key_and_desired_dim
        self.batch_size = batch_size
        self.chunk_size = chunk_size

        self.buffer = {k: List() for k in self.deskeydim.keys()}
        self.left = {k: [] for k in self.deskeydim.keys()}
        self.sample_key = next(iter(key_and_desired_dim.keys()))

        self.leftoverarray = lambda data: leftoverarray(data, self.chunk_size)
        self.getarrays = lambda data: getarrays(data, self.chunk_size)

    def push(self, data: Dict[str, jnp.ndarray]):
        data = tree_map(transform2ds, data, self.deskeydim)
        prechunk = tree_map(self.pusharray, data, self.left)
        self.left = {k: [] for k in self.deskeydim.keys()}

        chunks_tree = tree_map(self.getarrays, prechunk)
        left = tree_map(self.leftoverarray, prechunk)

        tree_map(lambda buffer, chunks: jax.vmap(lambda chunk: buffer.append(chunk))(chunks)
                 , self.buffer, chunks_tree)

        self.left = tree_map(self.pusharray, left, self.left)

    def sample(self, key, device=None):
        idx = int(jax.random.randint(key, shape=(), minval=0, maxval=len(self.buffer[self.sample_key])))
        data = tree_map(
            lambda data, expected_dim: transform2batch(data[idx], expected_dim, self.chunk_size, self.batch_size),
            self.buffer, self.deskeydim)
        data = tree_map(self.poparray, data,
                        {k: jax.devices()[0] if device is None else device for k in self.deskeydim.keys()})
        return data

    def pusharray(self, data, leftover):
        data_cpu = jax.device_put(data, device=jax.devices("cpu")[0])
        if len(leftover) == 0:
            return data_cpu
        else:
            return jax.lax.concatenate([leftover, data_cpu], 0)

    def poparray(self, data, device):
        return jax.device_put(data, device)


if __name__ == "__main__":
    import time

    rb = ReplayBuffer(1_000_000, {"obs": 4, "action": 2}, 16)
    elapsed_times = []
    for i in range(500):
        a = time.time()
        rb.push({"obs": jnp.zeros((2000, 64, 64, 3)), "action": jnp.zeros((2000, 6))})
        b = time.time() - a
        elapsed_times.append(b)
        # print(f"num of data pushes: {i}, num of current number of insertion of chunk: {rb.chunk_id}\n num of current number of chunk: {rb.buffer.keys()}\n num of ready to be merged into chunk: {rb.left['action'].shape}\n")
    print(f"average push time is:: {sum(elapsed_times) / len(elapsed_times)}")
    elapsed_time = []
    key = jax.random.key(0)
    datas = []
    for i in range(100):
        key, partial_key = jax.random.split(key, num=2)
        a = time.time()
        data = rb.sample(partial_key)
        b = time.time() - a
        # print(i, data.keys(), data['obs'].shape, data['obs'].device_buffer.device())
        datas.append(data)
        elapsed_times.append(b)
    print(f"average sample time is:: {sum(elapsed_times) / len(elapsed_times)}")
