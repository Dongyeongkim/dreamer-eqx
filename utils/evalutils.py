import numpy as np
import jax.numpy as jnp


def craftax_eval_fn(agent_fn, env_fn, key, agent_modules, env_params):
    report = {f"eval/{k}": np.array(v).mean() for k, v in env_fn.achievements.items()}
    report["eval/score"] = np.array(jnp.exp(jnp.mean(jnp.log(1 + jnp.array(list(report.values())))))) - 1.0
    return report
