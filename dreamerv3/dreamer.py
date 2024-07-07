import jax
import numpy as np
import equinox as eqx
from jax import random
import jax.numpy as jnp
from . import behaviors
from .models import WorldModel
from jax.tree_util import tree_map
from .dreamerutils import Moments
from .dreamerutils import SlowUpdater
from .dreamerutils import (
    tensorstats,
    balance_stats,
    add_colour_frame,
    get_feat,
    video_grid,
    MSEDist,
)


sg = lambda x: jax.tree_util.tree_map(jax.lax.stop_gradient, x)


def generate_dreamerV3_modules(key, obs_space, act_space, config):
    wm_key, ac_key, ac2_key = random.split(key, num=3)

    wm = WorldModel(wm_key, obs_space, act_space, config)
    task_behavior = getattr(behaviors, config.agent.task_behavior)(
        ac_key, act_space, config
    )
    if config.agent.expl_behavior == "None":
        expl_behavior = task_behavior
    else:
        expl_behavior = getattr(behaviors, config.agent.expl_behavior)(
            ac2_key, act_space, config
        )
    return {
        "wm": wm,
        "task_behavior": task_behavior,
        "expl_behavior": expl_behavior,
        "norms": {
            "retnorm": Moments(**config.agent.retnorm),
            "advnorm": Moments(**config.agent.advnorm),
            "valnorm": Moments(**config.agent.valnorm),
        },
        "updater": SlowUpdater(
            fraction=config.agent.slow_critic_fraction,
            period=config.agent.slow_critic_update,
        ),
    }


class DreamerV3:
    def __init__(self, obs_space, act_space, config=None):
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space
        self.scales = self.config.common.loss_scales

    def policy_initial(self, modules, batch_size):
        return (
            modules["wm"].initial(batch_size),
            modules["task_behavior"].initial(batch_size),
            modules["expl_behavior"].initial(batch_size),
        )

    def train_initial(self, modules, batch_size):
        return modules["wm"].initial(batch_size)

    @eqx.filter_jit
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
        return state, outs

    @eqx.filter_jit
    def train(self, modules, key, carry, data, opt, opt_state):
        context_data = data.copy()
        context = {
            k: context_data.pop(k)[:, :1] for k in modules["wm"].rssm.initial(1).keys()
        }
        context["stoch"] = jax.nn.one_hot(
            context["stoch"], self.config.wm.latent_cls
        ).astype(self.config.wm.cdtype)
        prevlat = modules["wm"].rssm.outs_to_carry(context)
        prevact = data["action"][:, 0]
        carry = prevlat, prevact
        data = {k: v[:, 1:] for k, v in data.items()}
        modules, opt_state, total_loss, loss_and_info = opt.update(
            key, opt_state, self.loss, modules, carry, data
        )

        # slow update for critic

        critic = eqx.filter(
            modules["task_behavior"].ac.critic["extr"].net, eqx.is_array
        )
        slowcritic, slowcritic_static = eqx.partition(
            modules["task_behavior"].ac.critic["extr"].slow, eqx.is_array
        )
        modules["updater"], ema_slowcritic = modules["updater"](critic, slowcritic)
        modules["task_behavior"].ac.critic["extr"] = eqx.tree_at(
            lambda mod: mod.slow,
            modules["task_behavior"].ac.critic["extr"],
            eqx.combine(ema_slowcritic, slowcritic_static),
        )

        return modules, opt_state, total_loss, loss_and_info

    def loss(self, modules, key, carry, data):
        losses = {}
        metrics = {}
        key, wm_loss_key, ac_loss_key = random.split(key, num=3)
        wm_losses, (wm_carry, wm_outs, wm_metrics) = modules["wm"].loss(
            wm_loss_key, carry, data
        )  # using wm_carry is available at the mode of 'last' mode. it will be added after few weeks.
        losses.update(wm_losses)
        metrics.update({f"train/{k}": v for k, v in wm_metrics.items()})
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
        ac_losses, modules["norms"], ac_metrics = modules["task_behavior"].loss(
            ac_loss_key, modules["norms"], modules["wm"].imagine, start
        )
        losses.update(ac_losses)
        metrics.update({f"train/{k}": v for k, v in ac_metrics.items()})

        if self.config.agent.replay_critic_loss:
            key, replay_ret_key = random.split(key, num=2)
            ret = losses.pop("ret")
            data_with_wm_outs = {**data, **wm_outs}
            replay_critic_loss, modules["norms"], replay_ret = (
                modules["task_behavior"]
                .ac.critic["extr"]
                .replay_critic_loss(modules["norms"], data_with_wm_outs, ret)
            )
            losses.update(replay_critic_loss)
            metrics.update(
                {
                    f"train/{k}": v
                    for k, v in tensorstats(
                        replay_ret_key, replay_ret, "replay_ret"
                    ).items()
                }
            )
        scaled_losses = {k: v * self.scales[k] for k, v in losses.items()}
        loss = jnp.stack([v.mean() for v in scaled_losses.values()]).sum()
        metrics.update({f"train/{k}_loss": v.mean() for k, v in scaled_losses.items()})
        metrics.update(
            {f"train/{k}_loss_std": v.std() for k, v in scaled_losses.items()}
        )
        newlat = wm_carry
        newact = data["action"][:, -1]
        newcarry = (newlat, newact)
        _, _ = data_with_wm_outs.pop("post"), data_with_wm_outs.pop("prior")
        data_with_wm_outs["stoch"] = jnp.argmax(data_with_wm_outs["stoch"], -1).astype(jnp.int32) # dreamer.py L106
        jax.debug.breakpoint()
        return loss, (modules["norms"], newcarry, data_with_wm_outs, metrics)

    @eqx.filter_jit
    def jax_report(self, modules, key, data):
        report = {}
        loss_key, obs_key, oloop_key = random.split(key, num=3)
        carry = modules["wm"].initial(len(data["is_first"]))

        # Loss and metrics
        losses, (loss_outs, carry_out, metrics) = modules["wm"].loss(
            loss_key, carry, data
        )
        report.update(metrics)
        report.update({f"{k}_loss": v.mean() for k, v in losses.items()})

        _, T = data["is_first"].shape
        num_obs = min(self.config.report.report_openl_context, T // 2)

        img_start, rec_outs = modules["wm"].rssm.observe(
            obs_key,
            carry[0],
            data["action"][:, :num_obs],
            eqx.filter_vmap(modules["wm"].encoder, in_axes=1, out_axes=1)(
                data["observation"][:, :num_obs]
            ),
            data["is_first"][:, :num_obs],
        )
        _, img_outs = modules["wm"].rssm.imagine(
            oloop_key, img_start, data["action"][:, num_obs:]
        )
        rec = dict(
            recon=MSEDist(
                eqx.filter_vmap(modules["wm"].heads["decoder"], in_axes=1, out_axes=1)(
                    rec_outs
                ),
                3,
                "sum",
            ),
            reward=modules["wm"].heads["reward"](get_feat(rec_outs)),
            cont=modules["wm"].heads["cont"](get_feat(rec_outs)),
        )
        img = dict(
            recon=MSEDist(
                eqx.filter_vmap(modules["wm"].heads["decoder"], in_axes=1, out_axes=1)(
                    img_outs
                ),
                3,
                "sum",
            ),
            reward=modules["wm"].heads["reward"](get_feat(img_outs)),
            cont=modules["wm"].heads["cont"](get_feat(img_outs)),
        )
        data_img = {k: v[:, num_obs:] for k, v in data.items()}
        data_img["recon"] = data_img["observation"]
        cont = data_img.pop("is_terminal")
        data_img["cont"] = 1 - jnp.float32(cont)
        losses = {k: -v.log_prob(data_img[k].astype("float32")) for k, v in img.items()}
        metrics.update({f"openl_{k}_loss": v.mean() for k, v in losses.items()})
        stats = balance_stats(img["reward"], data_img["reward"], 0.1)
        metrics.update({f"openl_reward_{k}": v for k, v in stats.items()})
        stats = balance_stats(img["cont"], data_img["cont"], 0.5)
        metrics.update({f"openl_cont_{k}": v for k, v in stats.items()})

        obs = rec["recon"].mode()[:6]
        openl = img["recon"].mode()[:6]
        true = data["observation"][:6]
        pred = jnp.concatenate([obs, openl], 1)
        error = (pred - true + 1) / 2
        model_w_grid = jnp.concatenate(
            [
                add_colour_frame(obs.astype("float32"), colour="green"),
                add_colour_frame(openl.astype("float32"), colour="red"),
            ],
            1,
        )
        video = jnp.concatenate([true, model_w_grid, error], 2)
        report[f"openl_decoder"] = video_grid(video)

        return report

    def report(self, modules, key, data):
        report = self.jax_report(modules, key, data)
        report = {f"report/{k}": np.float32(v) for k, v in report.items()}
        return report
