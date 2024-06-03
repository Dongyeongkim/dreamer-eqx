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

    def train(self):
        pass

    def loss(self):
        pass
