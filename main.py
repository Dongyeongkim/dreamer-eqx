import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["MUJOCO_RENDERER"] = "egl"
from envs import VecDmEnvWrapper, Walker2d, Cheetah

import hydra
import dreamerv3
import ml_collections
from jax import random
from jax import numpy as jp


def make_env(env_name: str, **kwargs) -> VecDmEnvWrapper:
    if env_name == "walker2d":
        env = Walker2d()
    elif env_name == "cheetah":
        env = Cheetah()
    else:
        raise ValueError(f"Unsupported environment: {env_name}")

    env = VecDmEnvWrapper(env, **kwargs)
    return env


def make_dreamer(env, config, key):
    obs_space = env.observation_spec()
    act_space = env.action_spec()
    dreamer = dreamerv3.DreamerV3(key, obs_space, act_space, config=config)
    return dreamer


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    config = ml_collections.ConfigDict(cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)

    key = random.key(config.seed)
    env = make_env(**config.env)
    dreamer = make_dreamer(env, config, key)
    dreamer_state = dreamer.policy_initial(config['env']['num_env'])

    for epoch in range(config.num_epoch):
        print(f"Epoch {epoch}")
        states = env.reset()
        for i in range(config.num_steps):
            dreamer_state, actions = dreamer.policy(key, dreamer_state, states)
            step_res = env.step(actions)
            if step_res.last():
                break
            key, _ = random.split(key)

    history = []
    states = env.reset()
    dreamer_state = dreamer.policy_initial(config['env']['num_env'])
    for i in range(100):
        dreamer_state, actions = dreamer.policy(key, states.observation, dreamer_state)
        states = env.step(actions)
        history.append(env.render())

    history = jp.stack(history)
    import mediapy as media
    import tempfile
    import webbrowser

    for i in range(3):
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".html") as f:
            url = "file://" + f.name
            f.write(media.show_video(history[:, i], return_html=True))
            webbrowser.open(url)


if __name__ == "__main__":
    main()
