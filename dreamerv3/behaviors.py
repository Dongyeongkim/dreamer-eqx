import jax
import equinox as eqx
import jax.numpy as jnp
from utils import OneHotDist, get_feat
from ml_collections import FrozenConfigDict
from models import ImagActorCritic, VFunction
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions


class Greedy(eqx.Module):
    ac: eqx.Module
    def __init__(self, wm, act_space, config):
        rewfn = lambda s: wm.heads["reward"](get_feat(s)).mean()[:, 1:]
        if config.critic_type == "vfunction":
            critics = {"extr": VFunction(rewfn, config, name="critic")}
        else:
            raise NotImplementedError(config.critic_type)
        self.ac = ImagActorCritic(critics, {"extr": 1.0}, act_space, config, name="ac")

    def initial(self, batch_size):
        return self.ac.initial(batch_size)

    def policy(self, latent, state):
        return self.ac.policy(latent, state)

    def train(self, imagine, start, data):
        return self.ac.train(imagine, start, data)

    def report(self, data):
        return {}


class Random(eqx.Module):
    act_space: any
    config: FrozenConfigDict
    def __init__(self, wm, act_space, config):
        self.config = config
        self.act_space = act_space

    def initial(self, batch_size):
        return jnp.zeros(batch_size)

    def policy(self, latent, state):
        batch_size = len(state)
        shape = (batch_size,) + self.act_space.shape
        if self.act_space.discrete:
            dist = OneHotDist(jnp.zeros(shape))
        else:
            dist = tfd.Uniform(-jnp.ones(shape), jnp.ones(shape))
            dist = tfd.Independent(dist, 1)
        return {"action": dist}, state

    def train(self, imagine, start, data):
        return None, {}

    def report(self, data):
        return {}
