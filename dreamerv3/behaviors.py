import equinox as eqx
from jax import random
import jax.numpy as jnp
from .models import ImagActorCritic, VFunction
from .dreamerutils import OneHotDist
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


class Greedy(eqx.Module):
    ac: eqx.Module

    def __init__(self, key, act_space, config):
        criticKey, acKey = random.split(key)
        if config.agent.critic_type == "vfunction":
            critics = {"extr": VFunction(criticKey, config)}
        else:
            raise NotImplementedError(config.agent.critic_type)
        self.ac = ImagActorCritic(acKey, critics, {"extr": 1.0}, act_space, config)

    def initial(self, batch_size):
        return self.ac.initial(batch_size)

    def policy(self, state, latent):
        return self.ac.policy(state, latent)

    def loss(self, key, norms, imagine, start):
        return self.ac.loss(key, norms, imagine, start)

    def report(self, data):
        return {}


class Random(eqx.Module):
    act_space: any

    def __init__(self, key, wm, act_space):
        self.act_space = act_space

    def initial(self, batch_size):
        return jnp.zeros(batch_size)

    def policy(self, state, latent):
        batch_size = len(state)
        shape = (batch_size,) + self.act_space.shape
        if self.act_space.discrete:
            dist = OneHotDist(jnp.zeros(shape))
        else:
            dist = tfd.Uniform(-jnp.ones(shape), jnp.ones(shape))
            dist = tfd.Independent(dist, 1)
        return state, {"action": dist}

    def report(self, data):
        return {}
