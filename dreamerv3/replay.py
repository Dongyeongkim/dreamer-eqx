import jax
import einops
import equinox as eqx
import jax.numpy as jnp
from jax.tree_util import tree_map

from typing import Dict


class ReplayBuffer:
    def __init__(self, buffer_size, key_and_desired_dim, batch_size, chunk_size=1024):
        assert chunk_size % batch_size == 0
        self.buffer_size = buffer_size
        self.deskeydim = key_and_desired_dim
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.chunk_id = 0

        self.buffer = {}
        self.left = {k: [] for k in self.deskeydim.keys()}

    def push(self, data: Dict[str, jnp.ndarray]):
        data = tree_map(self.transform2ds, data, self.deskeydim)
        prechunk = tree_map(self.pusharray, data, self.left)
        self.left = {k: [] for k in self.deskeydim.keys()}
        chunks_and_infos = tree_map(self.optimisedgetchunk, prechunk)
        chunk_cond = {k: v[0] for k, v in chunks_and_infos.items()}
        leftchunk_cond = [v[1] for v in chunks_and_infos.values()]
        numchunks = [v[2] for v in chunks_and_infos.values()]
        chunks = {k: v[3] for k, v in chunks_and_infos.items()}
        if True in chunk_cond.values():
            self.left = {k: self.optimisedgetleft(v) for k,v in chunks.items()}
        else:
            if leftchunk_cond[0]:
                for i in range(numchunks[0]):
                    idx = self.chunk_id % (self.buffer_size // self.chunk_size)
                    self.buffer.update({idx: {k: v[i] for k, v in chunks.items()}})
                    self.chunk_id += 1
                
                self.left = {k: self.optimisedgetleft(v) for k,v in chunks.items()}
                

            else:
                for i in range(numchunks[0]):
                    idx = self.chunk_id % (self.buffer_size // self.chunk_size)
                    self.buffer.update({idx: {k: v[i] for k, v in chunks.items()}})
                    self.chunk_id += 1
                
                self.left = {k: jnp.array([]) for k in chunks.keys()}

    def sample(self, key, device=None):
        if self.chunk_id > self.buffer_size // self.chunk_size:
            idx = int(
                jax.random.choice(key, jnp.arange(self.buffer_size // self.chunk_size))
            )
        else:
            idx = int(jax.random.choice(key, jnp.arange(self.chunk_id)))
        data = eqx.filter_jit(tree_map)(self.transform2batch, self.buffer[idx], self.deskeydim)
        data = tree_map(
            self.poparray,
            data,
            {
                k: jax.devices()[0] if device is None else device
                for k in self.deskeydim.keys()
            },
        )
        return data

    def pusharray(self, data, leftover):
        if len(leftover) == 0:
            return jax.device_put(data, device=jax.devices("cpu")[0])
        else:
            return jax.lax.concatenate(
                [leftover, jax.device_put(data, device=jax.devices("cpu")[0])], 0
            )

    def poparray(self, data, device):
        return jax.device_put(data, device)
        
    def optimisedgetchunk(self, data):
        return (
            ((len(data) % self.chunk_size) == len(data)),
            (len(data) % self.chunk_size),
            (len(data) // self.chunk_size),
            jnp.array_split(data, jnp.arange(self.chunk_size, len(data), self.chunk_size)),
        )
    
    def optimisedgetleft(self, data):
        return data[-1]
        
    
    def transform2ds(self, data: jnp.array, expected_dim: int):
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
    
    @eqx.filter_jit
    def transform2batch(self, data: jnp.array, expected_dim: int):
        assert (
            len(data.shape) == expected_dim
        ), "dimension does not fit with expected dim"
        return einops.rearrange(
            data,
            "(t b) ... -> b t ...",
            b=self.batch_size,
            t=self.chunk_size // self.batch_size,
        )


if __name__ == "__main__":
    import time

    rb = ReplayBuffer(1_000_000, {"obs": 4, "action": 2}, 16)
    elapsed_times = []
    for i in range(500):
        a = time.time()
        rb.push({"obs": jnp.zeros((2000, 64, 64, 3)), "action": jnp.zeros((2000, 6))})
        b = time.time() - a
        elapsed_times.append(b)
        print(
            f"num of data pushes: {i}, num of current number of insertion of chunk: {rb.chunk_id}\n num of current number of chunk: {len(rb.buffer.keys())}\n num of ready to be merged into chunk: {rb.left['action'].shape}\n"
        )
    print(f"average push time is:: {sum(elapsed_times) / len(elapsed_times)}")
    import matplotlib.pyplot as plt
    plt.plot(elapsed_times)
    plt.savefig("test-time.png")
    elapsed_time = []
    key = jax.random.key(0)
    datas = []
    for i in range(100):
        key, partial_key = jax.random.split(key, num=2)
        a = time.time()
        data = rb.sample(partial_key)
        b = time.time() - a
        print(i, data.keys(), data["obs"].shape, data["obs"].device_buffer.device())
        datas.append(data)
        elapsed_times.append(b)
    print(f"average sample time is:: {sum(elapsed_times) / len(elapsed_times)}")
    
