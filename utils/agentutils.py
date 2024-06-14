import dreamerv3


def make_dreamer(env, config, key):
    obs_space = env.observation_space(env.default_params).shape
    act_space = env.action_space(env.default_params).n
    dreamer = dreamerv3.DreamerV3(key, obs_space, act_space, config=config)
    modules = dreamerv3.generate_dreamerV3_modules(key, obs_space, act_space, config=config)
    return dreamer, modules
