import jax
from jax import random
import jax.numpy as jnp
from . import behaviors
from .models import WorldModel
from jax.tree_util import tree_map


sg = lambda x: jax.tree_util.tree_map(jax.lax.stop_gradient, x)


def generate_dreamerV3_modules(key, obs_space, act_space, config):
    wm_key, ac_key, ac2_key = random.split(key, num=3)

    wm = WorldModel(wm_key, obs_space, act_space, config)
    task_behavior = getattr(behaviors, config.agent.task_behavior)(
        ac_key, wm, act_space, config
    )
    if config.agent.expl_behavior == "None":
        expl_behavior = task_behavior
    else:
        expl_behavior = getattr(behaviors, config.agent.expl_behavior)(
            ac2_key, wm, act_space, config
        )
    return {
        "wm": wm,
        "task_behavior": task_behavior,
        "expl_behavior": expl_behavior,
    }


class DreamerV3:
    def __init__(self, key, obs_space, act_space, step=0, config=None):
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space
        self.step = step
        self.scales = self.config.loss_scales
        # self.opt = Optimizer(lr=self.config.lr)
        # self.opt_state = self.opt.init(self.modules)

    def policy_initial(self, modules, batch_size):
        return (
            modules["wm"].initial(batch_size),
            modules["task_behavior"].initial(batch_size),
            modules["expl_behavior"].initial(batch_size),
        )

    def train_initial(self, modules, batch_size):
        return modules["wm"].initial(batch_size)

    def policy(self, modules, key, state, obs, mode="train"):
        obs_key, act_key = random.split(key, num=2)
        embed = modules["wm"].encoder(obs["observation"])
        (prev_latent, prev_action), task_state, expl_state = state
        prev_latent["key"] = obs_key
        _, latent = modules["wm"].rssm.obs_step(
            prev_latent, (prev_action, embed, obs["is_first"])
        )
        _, _ = latent.pop("post"), latent.pop("prior")
        task_state, task_outs = modules["task_behavior"].policy(task_state, latent)
        expl_state, expl_outs = modules["expl_behavior"].policy(expl_state, latent)

        if mode == "eval":
            outs = task_outs
            outs["action"] = outs["action"].sample(seed=act_key)
            outs["log_entropy"] = jnp.zeros(outs["action"].shape[:1])
        elif mode == "explore":
            outs = expl_outs
            outs["log_entropy"] = outs["action"].entropy()
            outs["action"] = outs["action"].sample(seed=act_key)
        elif mode == "train":
            outs = task_outs
            outs["log_entropy"] = outs["action"].entropy()
            outs["action"] = outs["action"].sample(seed=act_key)
            state = ((latent, outs["action"]), task_state, expl_state)
        else:
            raise NotImplementedError
        return modules, state, outs

    def train(self, modules, key, carry, data, opt, opt_state):
        context_data = data.copy()
        context = {
            k: context_data.pop(k)[:, :1] for k in modules["wm"].rssm.initial(1).keys()
        }
        prevlat = modules["wm"].rssm.outs_to_carry(context)
        prevact = data["action"][:, 0]
        carry = prevlat, prevact
        data = {k: v[:, 1:] for k, v in data.items()}
        modules, opt_state, total_loss, loss_and_info = opt.update(
            key, opt_state, self.loss, modules, carry, data
        )
        return modules, total_loss, loss_and_info, opt_state

    def loss(self, modules, key, carry, data):
        losses = {}
        metrics = {}
        wm_loss_key, ac_loss_key = random.split(key, num=2)
        wm_losses, (wm_carry, wm_outs, wm_metrics) = modules["wm"].loss(
            wm_loss_key, carry, data
        )  # using wm_carry is available at the mode of 'last' mode. it will be added after few weeks.
        losses.update(wm_losses)
        metrics.update(wm_metrics)
        rew = data["reward"]
        con = 1 - jnp.float32(data["is_terminal"])
        B, T = data["is_first"].shape
        startlat = modules["wm"].rssm.outs_to_carry(
            tree_map(lambda x: x.reshape((B * T, 1, *x.shape[2:])), wm_outs)
        )
        startout, startrew, startcon = tree_map(
            lambda x: x.reshape((B * T, *x.shape[2:])), (wm_outs, rew, con)
        )
        startlat, startout, startrew, startcon = tree_map(
            lambda x: x.repeat(1, 0), (startlat, startout, startrew, startcon)
        )
        start = {
            "startlat": startlat,
            "startout": startout,
            "startrew": startrew,
            "startcon": startcon,
        }
        ac_losses, ac_metrics = modules["task_behavior"].loss(
            ac_loss_key, modules["wm"].imagine, start
        )
        losses.update(ac_losses)
        metrics.update(ac_metrics)

        if self.config.replay_critic_loss:
            ret = losses.pop("ret")
            data_with_wm_outs = {**data, **wm_outs}
            replay_critic_loss = (
                modules["task_behavior"]
                .ac.critic["extr"]
                .replay_critic_loss(data_with_wm_outs, ret)
            )
            losses.update(replay_critic_loss)

        scaled_losses = {k: v * self.scales[k] for k, v in losses.items()}
        loss = jnp.stack([v.mean() for v in scaled_losses.values()]).sum()

        return loss, (scaled_losses, metrics)
