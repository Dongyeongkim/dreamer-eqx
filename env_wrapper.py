import gym
import jax.numpy as jnp


class DMC_JAX:
    def __init__(self, env):
        self._env = env
        self._ignored_keys = []

    @property
    def obs_space(self):
        spaces = {
            "observation": gym.spaces.Box(0, 255, (64, 64, 3), dtype=jnp.float32),
            "reward": gym.spaces.Box(-jnp.inf, jnp.inf, (), dtype=jnp.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=jnp.float32),
            "is_last": gym.spaces.Box(0, 1, (), dtype=jnp.float32),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=jnp.float32),
        }
        return spaces

    @property
    def act_space(self):
        spec = self._env.action_spec()
        action = gym.spaces.Box(
            (spec.minimum) * spec.shape[0],
            (spec.maximum) * spec.shape[0],
            shape=spec.shape,
            dtype=jnp.float32,
        )
        return {"action": action}

    def step(self, action):
        time_step = self._env.step(action)
        # assert time_step.discount in (0, 1) disable this assertion due to vecotrise feaature
        obs = {
            "reward": time_step.reward,
            "is_first": jnp.zeros_like(self._env.num_env),
            "is_last": time_step.last(),
            "is_terminal": jnp.equal(time_step.discount, 0),
            "observation": time_step.observation,
            "action": action,
            "discount": time_step.discount,
        }
        return obs

    def reset(self):
        time_step = self._env.reset()
        obs = {
            "reward": 0.0,
            "is_first": jnp.ones_like(self._env.num_env),
            "is_last": jnp.zeros_like(self._env.num_env),
            "is_terminal": jnp.zeros_like(self._env.num_env),
            "observation": time_step.observation,
            "action": jnp.zeros_like(self.act_space["action"].sample()),
            "discount": time_step.discount,
        }
        return obs

    def __getattr__(self, name):
        if name == "obs_space":
            return self.obs_space
        if name == "act_space":
            return self.act_space
        return getattr(self._env, name)



