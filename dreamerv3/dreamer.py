from . import behaviors
from jax import random
import jax.numpy as jnp
from .utils import Optimizer, SlowUpdater
from .models import WorldModel



class DreamerV3:
    def __init__(self, key, obs_space, act_space, step=0, config=None):
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space
        self.step = step
        self.scales = self.config.loss_scales


        wm_key, ac_key, ac2_key = random.split(key, num=3)

        self.wm = WorldModel(wm_key, self.obs_space, self.act_space, self.config)
        self.task_behavior = getattr(behaviors, config.agent.task_behavior)(
            ac_key, self.wm, self.act_space, self.config
        )
        if config.agent.expl_behavior == "None":
            self.expl_behavior = self.task_behavior
        else:
            self.expl_behavior = getattr(behaviors, config.agent.expl_behavior)(
                ac2_key, self.wm, self.act_space, self.config
            )
        self.updater = SlowUpdater(fraction=self.config.slow_critic_fraction, period=self.config.slow_critic_fraction)
        self.modules = {"wm": self.wm, "ac": self.task_behavior}

    def policy_initial(self, batch_size):
        return (
            self.wm.initial(batch_size),
            self.task_behavior.initial(batch_size),
            self.expl_behavior.initial(batch_size),
        )

    def train_initial(self, batch_size):
        return self.wm.initial(batch_size)

    def policy(self, key, state, obs, mode="train"):
        obs_key, act_key = random.split(key, num=2)
        embed = self.wm.encoder(obs["observation"])
        (prev_latent, prev_action), task_state, expl_state = state
        prev_latent["key"] = obs_key
        _, latent = self.wm.rssm.obs_step(
            prev_latent, (prev_action, embed, obs["is_first"]))
        _, _ = latent.pop("post"), latent.pop("prior")
        task_state, task_outs = self.task_behavior.policy(task_state, latent)
        expl_state, expl_outs = self.expl_behavior.policy(expl_state, latent)

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

    def train(self, key, carry, data):
        wm_loss_key, ac_loss_key = random.split(key, num=2)

    def loss(self, key, carry, data):
        wm_loss_key, ac_loss_key = random.split(key, num=2)
        wm_losses, (wm_carry, wm_outs, wm_metrics) = self.modules["wm"].loss(wm_loss_key, carry, data)
        ac_losses, ac_metrics = self.modules["ac"].loss(ac_loss_key, self.modules["wm"].imagine, wm_carry)
        return {**wm_losses, **ac_losses}