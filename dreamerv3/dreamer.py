import behaviors
import equinox as eqx
from models import WorldModel



class DreamerV3:
    def __init__(self, key, obs_space, act_space, step, config):
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space['action']
        self.step = step
        self.wm = WorldModel()
        self.task_behavior = getattr(behaviors, config.task_behavior)(self.wm, self.act_space, self.config)
        if config.expl_behavior == 'None':
            self.expl_behavior = self.task_behavior
        else:
            self.expl_behavior = getattr(behaviors, config.expl_behavior)(self.wm, self.act_space, self.config)


    def train(self):
        pass


    def loss(self):
        pass


        

    
