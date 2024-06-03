import behaviors
import equinox as eqx
from jax import random
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

    def init_policy(self, batch_size):
        pass

    def init_train(self, batch_size):
        pass


    def policy(self):
        pass

    def train(self, key, data, state):
        wm_loss_key, ac_loss_key = random.split(key, num=2)
        pass


    def loss(self, key, data, state):
        wm_loss_key, ac_loss_key = random.split(key, num=2)
        wm_loss, (wm_carry, wm_outs, wm_metrics) = self.wm.loss(wm_loss_key, data, state)
        ac_loss, ac_metrics = self.task_behavior.loss(ac_loss_key, self.wm.imagine, wm_carry)