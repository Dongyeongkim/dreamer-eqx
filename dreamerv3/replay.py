import jax
import einops
import jax.numpy as jnp
from collections import deque
from jax.tree_util import tree_map

from typing import Dict



class ReplayBuffer:
    def __init__(self, desireable_key_and_dim, chunk_size=1024):
        self.deskeydim = desireable_key_and_dim
        self.chunk_size = chunk_size
        self.num_chunk = 0

        self.buffer = {}
        self.left = {}

    def push(self, data: Dict[str, jnp.ndarray]):
        data = tree_map(lambda x, y: self.transform2ds(x, y), (data, self.deskeydim))
        for k, v in data.items():
            if k in self.left and self.left[k] is not None:
                chunk = jax.device_put(jnp.concatenate([self.left[k], v], axis=0), device=jax.devices("cpu")[0])
                self.left[k] = None
            else:
                chunk = jax.device_put(v, device=jax.devices("cpu")[0])

            if k not in self.buffer:
                self.buffer[k] = deque()

            while len(chunk) >= self.chunk_size:
                self.buffer[k].put(chunk[:self.chunk_size])
                chunk = chunk[self.chunk_size:]

            if len(chunk) > 0:
                self.left[k] = chunk

    def pop(self):
        data = {}
        for k, v in self.buffer.items():
            if not v.empty():
                data[k] = jax.device_put(v.popleft(), device=jax.devices()[0])
            else:
                return None
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