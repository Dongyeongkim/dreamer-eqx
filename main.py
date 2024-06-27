import hydra


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):

    import os
    import ml_collections
    from omegaconf import OmegaConf

    config = ml_collections.ConfigDict(OmegaConf.to_container(cfg, resolve=True))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.accelerator.gpu_id)
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".90"
    path = hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]

    from utils.logger import Logger
    from utils.envutils import make_craftax_env
    from utils.agentutils import make_dreamer
    from utils.trainutils import prefill_fn, train_and_evaluate_fn
    from utils.evalutils import craftax_eval_fn
    from dreamerv3.replay import generate_replaybuffer

    logger = Logger(path)

    import jax
    from jax import random

    key = random.key(config.seed)

    print(f"Building the Environment...")

    env = make_craftax_env(**config.env)
    eval_env = make_craftax_env(env_name=config.env.env_name, autoreset=config.env.autoreset, eval_mode=True)

    print(f"Done!")

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
    print("ReplayBuffer is generating")

    rb_state = generate_replaybuffer(
        buffer_size=config.common.rb_size,
        desired_key_dim={
            "deter": (config.wm.deter,),
            "stoch": (config.wm.latent_dim, config.wm.latent_cls),
            "observation": env.observation_space(env_params).shape,
            "reward": (),
            "is_first": (),
            "is_last": (),
            "is_terminal": (),
            "action": (env.action_space(env_params).n,),
        },
        batch_size=config.common.batch_size,
        batch_length=config.common.batch_length,
        num_env=config.env.num_envs,
    )

    print("ReplayBuffer has set up")

    print("Prefilling steps...")

    key, prefill_key = jax.random.split(key)
    rb_state = prefill_fn(
        prefill_key,
        config.common.batch_size * config.common.batch_length,
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
    state = train_and_evaluate_fn(
        key=training_key,
        num_steps=int(config.env.num_interaction_steps//config.env.num_envs),
        defrag_ratio=config.common.batch_size * config.common.batch_length,
        replay_ratio=(config.env.replay_ratio//config.env.num_envs),
        debug_mode=config.report.debug_mode,
        report_ratio=config.report.report_ratio,
        eval_ratio=config.report.eval_ratio,
        logger=logger,
        agent_fn=dreamer,
        env_fn=env,
        opt_fn=opt_fn,
        eval_fn=craftax_eval_fn,
        eval_env_fn=eval_env,
        agent_modules=dreamer_modules,
        agent_state=dreamer_state,
        env_params=env_params,
        env_state=env_state,
        opt_state=opt_state,
        rb_state=rb_state,
    )


if __name__ == "__main__":
    main()
