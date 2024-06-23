import dreamerv3
from dreamerv3.dreamerutils import Optimizer


def make_dreamer(env, config, key):
    obs_space = env.observation_space(env.default_params).shape
    act_space = env.action_space(env.default_params).n
    dreamer = dreamerv3.DreamerV3(obs_space, act_space, config=config)
    modules = dreamerv3.generate_dreamerV3_modules(
        key, obs_space, act_space, config=config
    )
    optim = Optimizer(lr=config.lr, scaler="rms", agc=0.3, eps=1e-20, momentum=True)
    opt_state = optim.init(modules)
    return dreamer, modules, optim, opt_state
