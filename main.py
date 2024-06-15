import os
import hydra
import ml_collections

from utils.envutils import make_craftax_env
from utils.agentutils import make_dreamer
from utils.trainutils import rollout_fn, prefill_fn
from dreamerv3.replay import generate_replaybuffer


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
    key, reset_key = jax.random.split(key)

    print(f"Setup the environments...")

    env_state, obs = env.reset(reset_key, env_params)

    print(f"Setting the environments is now done...")
    print("Building the agent...")

    key, dreamer_key = jax.random.split(key)
    dreamer, dreamer_modules, opt_fn, opt_state = make_dreamer(env, config, dreamer_key)
    # need to add optimiser state; which is from the optimiser side
    dreamer_state = dreamer.policy_initial(dreamer_modules, config.env.num_envs)

    print("Building the agent is now done...")
    print("ReplayBuffer generation")

    rb_state = generate_replaybuffer(
        buffer_size=config.rb_size,
        desired_key_dim={
            "deter": (config.rssm.deter,),
            "stoch": (config.rssm.latent_dim, config.rssm.latent_cls),
            "observation": (63, 63, 3),
            "reward": (),
            "is_first": (),
            "is_last": (),
            "is_terminal": (),
            "action": (17,),
        },
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        num_env=config.env.num_envs,
    )

    print("Prefilling steps...")

    key, prefill_key = jax.random.split(key)
    rb_state = prefill_fn(
        prefill_key,
        65,
        dreamer,
        env,
        opt_fn,
        dreamer_modules,
        dreamer_state,
        env_params,
        env_state,
        opt_state,
        rb_state,
    )
    print("Prefilled!")
    key, training_key = jax.random.split(key)
    state = rollout_fn(
        key=training_key,
        num_steps=65*30,
        defrag_ratio=65,
        replay_ratio=2,
        agent_fn=dreamer,
        env_fn=env,
        opt_fn=opt_fn,
        agent_modules=dreamer_modules,
        agent_state=dreamer_state,
        env_params=env_params,
        env_state=env_state,
        opt_state=opt_state,
        rb_state=rb_state)
    
    breakpoint()


if __name__ == "__main__":
    main()
