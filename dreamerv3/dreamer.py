import behaviors
import equinox as eqx
from jax import random
import jax.numpy as jnp
from models import WorldModel


class DreamerV3:
    def __init__(self, key, obs_space, act_space, step, config):
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space["action"]
        self.step = step
        wm_key, ac_key, ac2_key = random.split(key, num=2)

        self.wm = WorldModel(wm_key, self.obs_space, self.act_space, self.config)
        self.task_behavior = getattr(behaviors, config.task_behavior)(
            ac_key, self.wm, self.act_space, self.config
        )
        if config.expl_behavior == "None":
            self.expl_behavior = self.task_behavior
        else:
            self.expl_behavior = getattr(behaviors, config.expl_behavior)(
                ac2_key, self.wm, self.act_space, self.config
            )

    def policy_initial(self, batch_size):
        return (
            self.wm.initial(batch_size),
            self.task_behavior.initial(batch_size),
            self.expl_behavior.initial(batch_size),
        )

    def train_initial(self, batch_size):
        return self.wm.initial(batch_size)

    def policy(self, key, obs, state, mode="train"):
        obs_key, act_key = random.split(key, num=2)
        embed = self.wm.encoder(obs["image"])
        (prev_latent, prev_action), task_state, expl_state = state
        prev_latent["key"] = obs_key
        _, latent = self.wm.rssm.obs_step(
            prev_latent, (prev_action, embed, obs["is_first"])
        )
        self.expl_behavior.policy(expl_state, latent)
        task_outs, latent = self.task_behavior.policy(task_state, latent)
        expl_outs, latent = self.expl_behavior.policy(expl_state, latent)

        if mode == "eval":
            outs = task_outs
            outs["action"] = outs["action"].sample(seed=act_key)
            outs["log_entropy"] = jnp.zeros(outs["action"].shape[:1])
        elif mode == "explore":
            outs = expl_outs
            outs["log_entropy"] = outs["action"].entropy()
            outs["action"] = outs["action"].sample(seed=act_key)
        elif mode == "train":
            outs = task_outs
            outs["log_entropy"] = outs["action"].entropy()
            outs["action"] = outs["action"].sample(seed=act_key)
            state = ((latent, outs["action"]), task_state, expl_state)
        return state, outs

    def train(self, key, data, state):
        wm_loss_key, ac_loss_key = random.split(key, num=2)
        pass

    def loss(self, key, data, state):
        wm_loss_key, ac_loss_key = random.split(key, num=2)
        wm_loss, (wm_carry, wm_outs, wm_metrics) = self.wm.loss(
            wm_loss_key, data, state
        )
        ac_loss, ac_metrics = self.task_behavior.loss(
            ac_loss_key, self.wm.imagine, wm_carry
        )
        pass
