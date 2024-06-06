from queue import Queue
from typing import Dict

import jax.numpy as jp


class ReplayBuffer:
    def __init__(self, chunk_size=1024):
        self.chunk_size = chunk_size

        self.buffer = {}
        self.left = {}

    def push(self, data: Dict[str, jp.ndarray]):
        for k, v in data.items():
            if k in self.left and self.left[k] is not None:
                data = jp.concatenate([self.left[k], v], axis=0)
                self.left[k] = None
            else:
                data = v

            if k not in self.buffer:
                self.buffer[k] = Queue()

            while len(data) >= self.chunk_size:
                self.buffer[k].put(data[:self.chunk_size])
                data = data[self.chunk_size:]

            if len(data) > 0:
                self.left[k] = data

    def pop(self):
        data = {}
        for k, v in self.buffer.items():
            if not v.empty():
                data[k] = v.get()
            else:
                return None
        return data
