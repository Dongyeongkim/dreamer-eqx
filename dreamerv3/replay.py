import jax
import einops
import equinox as eqx
import jax.numpy as jnp
from collections import deque
from jax.tree_util import tree_map

from typing import Dict



class ReplayBuffer:
    def __init__(self, buffer_size, key_and_desired_dim, chunk_size=1024):
        self.buffer_size = buffer_size
        self.deskeydim = key_and_desired_dim
        self.chunk_size = chunk_size
        self.chunk_id = 0

        self.buffer = {}
        self.left = {k: [] for k in self.deskeydim.keys()}


    @eqx.filter_jit
    def push(self, data: Dict[str, jnp.ndarray]):
        data = tree_map(self.transform2ds, data, self.deskeydim)
        prechunk = tree_map(self.pusharray, data, self.left)
        self.left = {k: [] for k in self.deskeydim.keys()}
        while True:
            chunk = tree_map(self.getarray, prechunk)
            left = tree_map(self.leftoverarray, prechunk)
            if None in chunk.values():
                self.left = tree_map(self.pusharray, left, self.left)
                break
            else:
                if self.chunk_id * self.chunk_size > self.buffer_size:
                    self.chunk_id = 0
                self.buffer.update({self.chunk_id: chunk})
                self.chunk_id += 1
                prechunk = left

    def pop(self):
        data = {}
        for k, v in self.buffer.items():
            if not v.empty():
                data[k] = jax.device_put(v.popleft(), device=jax.devices()[0])
            else:
                return None
        return data
    
    def pusharray(self, data, leftover):
        if len(leftover) == 0:
            return jax.device_put(data, device=jax.devices("cpu")[0])
        else:
            return jax.device_put(jnp.concatenate([leftover, data], axis=0), device=jax.devices("cpu")[0])
    
    def getarray(self, data):
        if len(data) >= self.chunk_size:
            return data[:self.chunk_size]
        else:
            return None
        
    def leftoverarray(self, data):
        if len(data) > self.chunk_size:
            return data[self.chunk_size:]
        elif len(data) == self.chunk_size:
            return []
        else:
            return data

    
    def transform2ds(self, data: jnp.array, expected_dim: int):
        # IMPORTANT: EXPECTED SHAPE -> (T, B, ...)
        assert len(data.shape) < expected_dim + 2, " dimension cannot be bigger than expected shape + batch dimension"
        assert len(data.shape) > expected_dim - 1, " dimension cannot be smaller than expected shape"
        if len(data.shape) == expected_dim:
            return data
        elif len(data.shape) == expected_dim + 1:
            return einops.rearrange(data, 't b ... -> (t b) ...')
        else:
            raise NotImplementedError("Something is wrong")
    
    def transform2batch(self, data: jnp.array, expected_dim: int, batch_size: int):
        assert len(data.shape) == expected_dim, "dimension does not fit with expected dim"
        assert self.chunk_size % batch_size == 0
        return einops.rearrange(data, '(t b) ... -> b t ...', b=batch_size, t=self.chunk_size//batch_size)



if __name__ == "__main__":
    import time
    rb = ReplayBuffer(5_000_000, {"obs": 4, "action": 2})
    elapsed_times = []
    for _ in range(100):
        a = time.time()
        rb.push({"obs": jnp.zeros((1000, 64, 64, 3)), "action": jnp.zeros((1000, 6))})
        b = time.time() - a
        elapsed_times.append(b)
    print(f"average push time is:: {sum(elapsed_times) / len(elapsed_times)}")