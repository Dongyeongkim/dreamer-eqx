import os
import hydra
import ml_collections

from utils.envutils import make_craftax_env
from utils.agentutils import make_dreamer
from utils.trainutils import rollout_fn


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    config = ml_collections.ConfigDict(cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    import jax
    from jax import random
    key = random.key(config.seed)
    print(f"Environment is compiling...")
    env = make_craftax_env(**config.env)
    print(f"Compiling is done...")
    env_params = env.default_params
    rng, reset_key = jax.random.split(key)
    print(f"Setup the environments...")
    first_obs, env_state = env.reset(reset_key, env_params)
    print(f"Setting the environments is now done...")
    print("Building the agent...")
    key, dreamer_key = jax.random.split(key)
    dreamer, modules = make_dreamer(env, config, dreamer_key)
    dreamer_state = dreamer.policy_initial(modules, config["env"]["num_envs"])
    print("Building the agent is now done...")
    print("ReplayBuffer generation")

if __name__ == "__main__":
    main()
