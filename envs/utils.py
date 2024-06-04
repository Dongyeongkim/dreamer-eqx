from typing import Optional

from brax.envs.base import PipelineEnv
from brax.io import image
import dm_env
from dm_env import specs
import jax
from jax import numpy as jp


class VecDmEnvWrapper(dm_env.Environment):
    def __init__(
        self,
        env: PipelineEnv,
        seed: int = 0,
        backend: Optional[str] = None,
        num_env: int = 1,
        width: int = 256,
        height: int = 256,
        use_image: bool = False,
    ):
        self._env = env
        self.seed(seed)
        self.backend = backend
        self._states = None
        self.num_env = num_env

        self.width = width
        self.height = height
        self.use_image = use_image

        if hasattr(self._env, "observation_spec"):
            self._observation_spec = self._env.observation_spec()
        else:
            obs_high = jp.inf * jp.ones(self._env.observation_size, dtype="float32")
            self._observation_spec = specs.BoundedArray(
                (self._env.observation_size,),
                minimum=-obs_high,
                maximum=obs_high,
                dtype="float32",
                name="observation",
            )

        if hasattr(self._env, "action_spec"):
            self._action_spec = self._env.action_spec()
        else:
            action_high = jp.ones(self._env.action_size, dtype="float32")
            self._action_spec = specs.BoundedArray(
                (self._env.action_size,),
                minimum=-action_high,
                maximum=action_high,
                dtype="float32",
                name="action",
            )

        self._reward_spec = specs.Array(shape=(), dtype="float32", name="reward")
        self._discount_spec = specs.BoundedArray(
            shape=(), dtype="float32", minimum=0.0, maximum=1.0, name="discount"
        )
        if hasattr(self._env, "discount_spec"):
            self._discount_spec = self._env.discount_spec()

        def reset(key):
            key1, *keys = jax.random.split(key, num=self.num_env + 1)
            keys = jax.numpy.array(keys)
            states = jax.vmap(lambda x: self._env.reset(x))(keys)
            return states, jax.vmap(lambda x: x.obs)(states), key1

        self._reset = jax.jit(reset, backend=self.backend)

        def step(states, action):
            states = jax.vmap(lambda x, y: self._env.step(x, y))(states, action)
            info = jax.vmap(lambda x: {**x.metrics, **x.info})(states)
            return (
                states,
                jax.vmap(lambda x: x.obs)(states),
                jax.vmap(lambda x: x.reward)(states),
                jax.vmap(lambda x: x.done)(states),
                info,
            )

        self._step = jax.jit(step, backend=self.backend)

    def reset(self):
        self._states, obs, self._key = self._reset(self._key)
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=None,
            discount=jp.ones(dtype="float32", shape=(self.num_env,)),
            observation=self.render() if self.use_image else obs,
        )

    def step(self, action):
        self._states, obs, reward, done, info = self._step(self._states, action)
        del info
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID if not done.any() else dm_env.StepType.LAST,
            reward=reward,
            discount=jp.ones(dtype="float32", shape=(self.num_env,)),
            observation=self.render() if self.use_image else obs,
        )

    def seed(self, seed: int = 0):
        self._key = jax.random.key(seed)

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def reward_spec(self):
        return self._reward_spec

    def discount_spec(self):
        return self._discount_spec

    def render(self):
        sys, states = self._env.sys, self._states
        if states is None:
            raise RuntimeError("must call reset or step before rendering")

        class VState:
            def __init__(self, q, qd):
                self.q = q
                self.qd = qd

        q = states.pipeline_state.q
        qd = states.pipeline_state.qd
        return jax.numpy.array(
            image.render_array(
                sys,
                [VState(q[i], qd[i]) for i in range(self.num_env)],
                self.width,
                self.height,
            )
        )


if __name__ == "__main__":
    from dmc_suite_brax.walker import Walker2d

    num_env = 16
    env = Walker2d()
    env = VecDmEnvWrapper(env, num_env=num_env)
    states = env.reset()
    history = []
    for i in range(100):
        print(f"step {i}")
        actions = jax.random.uniform(
            jax.random.key(i), (num_env, *env.action_spec().shape)
        )
        states = env.step(actions)
        history.append(env.render())
        if states.step_type == dm_env.StepType.LAST:
            states = env.reset()

    history = jp.stack(history)

    import mediapy as media
    import tempfile
    import webbrowser

    for i in range(3):
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".html") as f:
            url = "file://" + f.name
            f.write(media.show_video(history[:, i], return_html=True))
            webbrowser.open(url)
