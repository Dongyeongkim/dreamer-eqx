import jax
import numpy as np
import equinox as eqx
from jax import random
import jax.numpy as jnp
from .utils import (
    MSEDist,
    Moments,
    get_feat,
    tensorstats,
    subsample,
    image_grid,
    add_colour_frame,
    SlowUpdater,
)
from .networks import RSSM, ImageEncoder, ImageDecoder, MLP
from ml_collections import FrozenConfigDict
from typing import Callable

sg = lambda x: jax.tree_util.tree_map(jax.lax.stop_gradient, x)


class WorldModel(eqx.Module):
    rssm: eqx.Module
    encoder: eqx.Module
    heads: dict
    obs_space: tuple
    act_space: int
    config: FrozenConfigDict

    def __init__(self, key, obs_space, act_space, config):
        encoder_param_key, rssm_param_key, heads_param_key = random.split(key, num=3)
        self.obs_space = obs_space
        self.act_space = act_space
        self.encoder = ImageEncoder(encoder_param_key, **config.encoder)
        self.rssm = RSSM(rssm_param_key, action_dim=act_space, **config.rssm)
        dec_param_key, rew_param_key, cont_param_key = random.split(
            heads_param_key, num=3
        )
        self.heads = {
            "decoder": ImageDecoder(
                dec_param_key,
                **config.decoder,
            ),
            "reward": MLP(
                rew_param_key,
                out_shape=(),
                **config.reward_head,
            ),
            "cont": MLP(
                cont_param_key,
                out_shape=(),
                **config.cont_head,
            ),
        }
        self.config = FrozenConfigDict(config)

    def initial(self, batch_size):
        prev_latent = self.rssm.initial(batch_size)
        prev_action = jnp.zeros(
            (batch_size, self.act_space)
        )  # act_space should be integer
        return prev_latent, prev_action

    def loss(self, key, carry, data):
        step_key, loss_key = random.split(key, num=2)
        embeds = eqx.filter_vmap(self.encoder, in_axes=1, out_axes=1)(data["image"])
        prev_latent, prev_action = carry
        prev_actions = jnp.concatenate(
            [prev_action[:, None, ...], data["action"][:, :-1, ...]], 1
        )
        carry, outs = self.rssm.observe(
            step_key, prev_latent, prev_actions, embeds, data["is_first"]
        )
        losses, metrics = self.rssm.loss(loss_key, outs)
        for name, head in self.heads.items():
            log_name = name
            data_name = name
            if data_name == "decoder":
                log_name = "recon"
                data_name = "image"
                dist = eqx.filter_vmap(head, in_axes=1, out_axes=1)(outs)
                dist = MSEDist(dist.astype("float32"), 3, "sum")
            else:
                feat = get_feat(outs)
                dist = head(feat)
            losses.update({log_name: -dist.log_prob(data[data_name].astype("float32"))})
            if self.config.contdisc:
                del losses["cont"]
                softlabel = data["cont"] * (1 - 1 / self.config.discount_horizon)
                losses["cont"] = -dist["cont"].log_prob(softlabel)

        return losses, (carry, outs, metrics)

    def imagine(self, key, policy, start, horizon):
        first_cont = (1.0 - start["is_terminal"]).astype("float32")
        keys = list(self.rssm.initial(1).keys())
        start = {k: v for k, v in start.items() if k in keys}
        wm_key, policy_key = random.split(key, num=2)
        policy_key, policy_start_key = random.split(policy_key, num=2)
        start["key"] = wm_key
        start["policy_key"] = policy_key
        start["action"] = policy(policy_start_key, start)

        def step(prev, _):
            prev = prev.copy()
            carry, _ = self.rssm.img_step(prev, prev.pop("action"))
            policy_key, partial_key = random.split(prev["policy_key"], num=2)
            action = policy(partial_key, carry)
            return {**carry, "policy_key": policy_key, "action": action}, {
                **carry,
                "policy_key": policy_key,
                "action": action,
            }

        carry, traj = jax.lax.scan(
            f=lambda *a, **kw: step(*a, **kw), init=start, xs=jnp.arange(horizon)
        )
        _, _ = start.pop("key"), traj.pop("key")
        _, _ = start.pop("policy_key"), traj.pop("policy_key")

        traj = {
            k: jnp.concatenate([start[k][None], v], 0).swapaxes(1, 0)
            for k, v in traj.items()
        }
        cont = self.heads["cont"](self.rssm.get_feat(traj)).mode()
        traj["cont"] = jnp.concatenate([first_cont, cont[:, 1:]], 1)
        discount = 1 if self.config.contdisc else 1 - 1 / self.config.agent.horizon
        traj["weight"] = jnp.cumprod(discount * traj["cont"], 1) / discount
        return traj

    @eqx.filter_jit
    def jax_report(self, key, data):
        report = {}
        loss_key, obs_key, oloop_key, img_key = random.split(key, num=4)
        state = self.initial(len(data["is_first"]))
        losses, metrics = self.loss(loss_key, data, state)
        report.update({f"{k}_loss": v.sum(axis=-1).mean() for k, v in losses.items()})
        report.update(metrics)
        carry, outs = self.rssm.observe(
            obs_key,
            self.rssm.initial(8),
            data["action"][:8, ...],
            eqx.filter_vmap(self.encoder, in_axes=1, out_axes=1)(data["image"])[
                :8, ...
            ],
            data["is_first"][:8, ...],
        )

        truth = data["image"][:8].astype("float32")
        full_recon = eqx.filter_vmap(self.heads["decoder"], in_axes=1, out_axes=1)(
            outs
        ).astype("float32")
        error = (full_recon - truth + 1) / 2
        recon_video = jnp.concatenate(
            [truth, add_colour_frame(full_recon, colour="green"), error], 2
        )

        report[f"recon_video"] = recon_video

        carry, outs = self.rssm.observe(
            oloop_key,
            self.rssm.initial(8),
            data["action"][:8, :5, ...],
            eqx.filter_vmap(self.encoder, in_axes=1, out_axes=1)(data["image"])[
                :8, :5, ...
            ],
            data["is_first"][:8, :5, ...],
        )

        recon = eqx.filter_vmap(self.heads["decoder"], in_axes=1, out_axes=1)(
            outs
        ).astype("float32")

        _, states = self.rssm.imagine(img_key, carry, data["action"][:8, 5:, ...])

        openl = eqx.filter_vmap(self.heads["decoder"], in_axes=1, out_axes=1)(
            states
        ).astype("float32")
        model = jnp.concatenate([recon[:, :5], openl], 1)
        error = (model - truth + 1) / 2
        model_w_grid = jnp.concatenate(
            [
                add_colour_frame(recon[:, :5], colour="green"),
                add_colour_frame(openl, colour="red"),
            ],
            1,
        )

        video = jnp.concatenate([truth, model_w_grid, error], 2)
        report[f"openl_video"] = video
        report[f"openl_image"] = image_grid(video)

        return report

    def report(self, key, data):
        report = self.jax_report(key, data)
        report = {k: np.float32(v) for k, v in report.items()}

        return report


class ImagActorCritic(eqx.Module):
    actor: eqx.Module
    critic: eqx.Module
    retnorm: eqx.Module
    advnorm: eqx.Module

    config: FrozenConfigDict

    def __init__(self, key, critics, scales, act_space, config):
        self.actor = MLP(
            key,
            **config.agent.actor,
            out_shape=(act_space,) if isinstance(act_space, int) else act_space,
        )
        self.critic = critics
        self.retnorm = Moments(**config.agent.retnorm)
        self.advnorm = Moments(**config.agent.advnorm)
        self.config = FrozenConfigDict(config)

    def initial(self, batch_size):
        return {}

    def policy(self, carry, latent):
        return carry, {"action": self.actor(get_feat(latent))}

    def loss(self, key, imagine, start, update=True):
        metrics = {}
        losses = {}
        policy = lambda k, s: self.actor(sg(get_feat(s))).sample(seed=k)
        traj = imagine(key, policy, start, self.config.agent.imag_horizon)
        traj = {k: sg(v) for k, v in traj.items()}
        rew, ret, tarval, critic, slowcritic = self.critic.score(traj)

        voffset, vscale = self.critic.valnorm(ret, update)
        roffset, rscale = self.retnorm(ret, update)
        adv = (ret - tarval[:, :-1]) / rscale
        aoffset, ascale = self.advnorm(adv, update)
        adv_normed = (adv - aoffset) / ascale

        ret_normed = (ret - voffset) / vscale
        ret_padded = jnp.concatenate([ret_normed, 0 * ret_normed[:, -1:]], 1)
        losses["critic_loss"] = (
            traj["weight"][:, :-1]
            * -(
                critic.log_prob(sg(ret_padded))
                + self.config.agent.slowreg * critic.log_prob(sg(slowcritic.mean()))
            )[:, :-1]
        )

        actor = self.actor(get_feat(traj))

        logpi = sum(
            [v.log_prob(sg(traj["action"][k]))[:, :-1] for k, v in actor.items()]
        )
        ents = {k: v.entropy()[:, :-1] for k, v in actor.items()}
        losses["actor_loss"] = traj["weight"][:, :-1] * -(
            logpi * sg(adv_normed) + self.config.agent.actent * sum(ents.values())
        )
        return losses, metrics

    def _metrics(self, key, traj, policy, logpi, ent, adv):
        pass


class VFunction(eqx.Module):
    net: eqx.Module
    slow: eqx.Module
    valnorm: eqx.Module
    updater: eqx.Module
    rewfn: Callable
    config: FrozenConfigDict

    def __init__(self, key, rewfn, config):
        net_key, slow_key = random.split(key, num=2)
        self.net = MLP(net_key, out_shape=(), **config.agent.critic)
        self.slow = MLP(slow_key, out_shape=(), **config.agent.critic)
        self.valnorm = Moments(**config.agent.valnorm)
        self.updater = eqx.nn.Identity()
        self.rewfn = rewfn
        self.config = FrozenConfigDict(config)

    def score(self, traj, actor=None):
        rew = self.rewfn(traj)
        assert (
            len(rew) == len(traj["action"]) - 1
        ), "should provide rewards for all but last action"

        critic = self.net(traj)
        slowcritic = self.slow(traj)
        voffset, vscale = self.valnorm.stats()
        val = critic.mean() * vscale + voffset
        slowval = slowcritic.mean() * vscale + voffset
        tarval = slowval if self.config.agent.slowtar else val
        discount = 1 if self.config.contdisc else 1 - 1 / self.config.agent.horizon

        rets = [tarval[:, -1]]
        disc = traj["cont"][:, 1:] * discount
        lam = self.config.agent.return_lambda
        interm = rew[:, 1:] + (1 - lam) * disc * tarval[:, 1:]
        for t in reversed(range(disc.shape[1])):
            rets.append(interm[:, t] + disc[:, t] * lam * rets[-1])
        ret = jnp.stack(list(reversed(rets))[:-1], 1)
        return rew, ret, tarval, critic, slowcritic
