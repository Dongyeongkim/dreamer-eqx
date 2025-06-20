import jax
import jax.numpy as jnp
from functools import partial
from craftax.craftax_classic.constants import Achievement



class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


class BatchEnvWrapper(GymnaxWrapper):
    """Batches reset and step functions"""

    def __init__(self, env, num_envs: int):
        super().__init__(env)

        self.num_envs = num_envs

        self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, rng, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, env_state = self.reset_fn(rngs, params)
        return obs, env_state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, state, reward, done, info = self.step_fn(rngs, state, action, params)

        return obs, state, reward, done, info


class AutoResetEnvWrapper(GymnaxWrapper):
    """Provides standard auto-reset functionality, providing the same behaviour as Gymnax-default."""

    def __init__(self, env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, params=None):
        return self._env.reset(key, params)

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):

        rng, _rng = jax.random.split(rng)
        obs_st, state_st, reward, done, info = self._env.step(
            _rng, state, action, params
        )

        rng, _rng = jax.random.split(rng)
        obs_re, state_re = self._env.reset(_rng, params)

        # Auto-reset environment based on termination
        def auto_reset(done, state_re, state_st, obs_re, obs_st):
            state = jax.tree.map(
                lambda x, y: jax.lax.select(done, x, y), state_re, state_st
            )
            obs = jax.lax.select(done, obs_re, obs_st)

            return obs, state

        obs, state = auto_reset(done, state_re, state_st, obs_re, obs_st)

        return obs, state, reward, done, info


class OptimisticResetVecEnvWrapper(GymnaxWrapper):
    """
    Provides efficient 'optimistic' resets.
    The wrapper also necessarily handles the batching of environment steps and resetting.
    reset_ratio: the number of environment workers per environment reset.  Higher means more efficient but a higher
    chance of duplicate resets.
    """

    def __init__(self, env, num_envs: int, reset_ratio: int):
        super().__init__(env)

        self.num_envs = num_envs
        self.reset_ratio = reset_ratio
        assert (
            num_envs % reset_ratio == 0
        ), "Reset ratio must perfectly divide num envs."
        self.num_resets = self.num_envs // reset_ratio

        self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, rng, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, env_state = self.reset_fn(rngs, params)
        return obs, env_state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):

        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs_st, state_st, reward, done, info = self.step_fn(rngs, state, action, params)

        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_resets)
        obs_re, state_re = self.reset_fn(rngs, params)

        rng, _rng = jax.random.split(rng)
        reset_indexes = jnp.arange(self.num_resets).repeat(self.reset_ratio)

        being_reset = jax.random.choice(
            _rng,
            jnp.arange(self.num_envs),
            shape=(self.num_resets,),
            p=done,
            replace=False,
        )
        reset_indexes = reset_indexes.at[being_reset].set(jnp.arange(self.num_resets))

        obs_re = obs_re[reset_indexes]
        state_re = jax.tree.map(lambda x: x[reset_indexes], state_re)

        # Auto-reset environment based on termination
        def auto_reset(done, state_re, state_st, obs_re, obs_st):
            state = jax.tree.map(
                lambda x, y: jax.lax.select(done, x, y), state_re, state_st
            )
            obs = jax.lax.select(done, obs_re, obs_st)

            return state, obs

        state, obs = jax.vmap(auto_reset)(done, state_re, state_st, obs_re, obs_st)

        return obs, state, reward, done, info


class CraftaxWrapper(GymnaxWrapper):
    def __init__(self, env, shape=(64, 64)):
        self.achievements = {f"Achievements/{k.name.lower()}": [] for k in Achievement}
        self._shape = shape
        super().__init__(env)

    def reset(self, rng, params=None):
        obs, env_state = self._env.reset(rng, params)
        return env_state, {
            "observation": jax.image.resize(
                obs[None, ...],
                (1, self._shape[0], self._shape[1], obs.shape[2]),
                "nearest",
            ),
        }

    def step(self, rng, env_state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(
            rng, env_state, action, params
        )
        if info["discount"] == 0:
            for k in self.achievements.keys():                    
                self.achievements[k].append(int(info[k]))

        return env_state, {
            "observation": jax.image.resize(
                obs[None, ...],
                (1, self._shape[0], self._shape[1], obs.shape[2]),
                "nearest",
            ),
            "reward": reward,
            "is_first": jnp.bool(jnp.maximum(env_state.timestep, 0)),
            "is_last": done,
            "is_terminal": jnp.bool(info["discount"] == 0),
        }


class CraftaxEvalWrapper(GymnaxWrapper):
    def __init__(self, env, shape=(64, 64)):
        self._shape = shape
        super().__init__(env)

    def reset(self, rng, params=None):
        obs, env_state = self._env.reset(rng, params)
        return env_state, {
            "observation": jax.image.resize(
                obs,
                (obs.shape[0], self._shape[0], self._shape[1], obs.shape[3]),
                "nearest",
            ),
        }

    def step(self, rng, env_state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(
            rng, env_state, action, params
        )

        return env_state, {
            "observation": jax.image.resize(
                obs,
                (obs.shape[0], self._shape[0], self._shape[1], obs.shape[3]),
                "nearest",
            ),
            "reward": reward,
            "is_first": jnp.bool(jnp.maximum(env_state.timestep, 0)),
            "is_last": done,
            "is_terminal": jnp.bool(info["discount"] == 0),
            **info,
        }
