"""PPO for the wheeled-legged robot, tuned for throughput on a lower end GPUs.
some important notes:
  * Truncation is bootstrapped. When an episode ends because it hit the time limit
    rather than because the robot fell, the value of the true final state is added
    back into the reward. Treating a time limit as a terminal state teaches the
    policy that surviving to 20 s is worth zero, which quietly caps performance.
  * log_std is a state-independent parameter, not a network output. State-
    dependent std is a well-known way to get premature collapse of exploration.
  * Observations and value targets are normalised by running statistics.
"""

import os
import time
import math
from dataclasses import dataclass, asdict, field

import numpy as np
import torch
import torch.nn as nn

import robot_config as C
from vectorized import ThreadVecEnv


@dataclass
class PPOConfig:
    num_envs: int = 256
    rollout_steps: int = 24
    total_steps: int = 40_000_000
    num_epochs: int = 5
    num_minibatches: int = 4
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.005
    max_grad_norm: float = 1.0
    target_kl: float = 0.02  # early-stop the epoch loop, not a hard clamp
    hidden: tuple = (256, 256)
    init_log_std: float = -1.0
    anneal_lr: bool = True
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_threads: int = min(12, os.cpu_count() or 4)
    # logging / artefacts
    run_name: str = "ppo"
    log_every: int = 1
    save_every: int = 50
    record_every: int = 150  # video every N iterations; 0 disables
    record_seconds: float = 12.0
    out_dir: str = "runs"
    video_dir: str = "videos"

    @property
    def batch_size(self):
        return self.num_envs * self.rollout_steps

    @property
    def minibatch_size(self):
        return self.batch_size // self.num_minibatches


class RunningMeanStd(nn.Module):
    """Welford-style running statistics"""

    def __init__(self, shape, epsilon=1e-4):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(epsilon))

    @torch.no_grad()
    def update(self, x):
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot = self.count + batch_count
        self.mean += delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var.copy_(
            (m_a + m_b + delta.pow(2) * self.count * batch_count / tot) / tot
        )
        self.count.copy_(tot)

    def normalize(self, x):
        return (x - self.mean) * torch.rsqrt(self.var + 1e-8)

    def denormalize(self, x):
        return x * torch.sqrt(self.var + 1e-8) + self.mean


_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
_HALF_LOG_2PI_E = 0.5 * math.log(2.0 * math.pi * math.e)


def _hms(seconds):
    """Compact duration"""
    s = int(max(seconds, 0))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def mlp(sizes, out_gain):
    """Orthogonal-initialised MLP."""
    layers = []
    for i in range(len(sizes) - 1):
        layer = nn.Linear(sizes[i], sizes[i + 1])
        gain = out_gain if i == len(sizes) - 2 else np.sqrt(2.0)
        nn.init.orthogonal_(layer.weight, gain)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        if i < len(sizes) - 2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Gaussian policy + value function.

    The Normal distribution is written out by hand because it was faster.
    """

    def __init__(
        self, obs_dim=C.OBS_DIM, act_dim=C.ACT_DIM, hidden=(256, 256), init_log_std=-1.0
    ):
        super().__init__()
        self.actor = mlp([obs_dim, *hidden, act_dim], out_gain=0.01)
        self.critic = mlp([obs_dim, *hidden, 1], out_gain=1.0)
        # State-independent exploration noise.
        self.log_std = nn.Parameter(torch.full((act_dim,), float(init_log_std)))

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    def forward(self, obs):
        mean = self.actor(obs)
        return mean, self.log_std.expand_as(mean).exp(), self.critic(obs).squeeze(-1)

    def act(self, obs):
        mean = self.actor(obs)
        log_std = self.log_std
        z = torch.randn_like(mean)
        action = mean + z * log_std.exp()
        logprob = -(0.5 * z * z + log_std + _LOG_SQRT_2PI).sum(-1)
        return action, logprob, self.critic(obs).squeeze(-1)

    def evaluate_actions(self, obs, action):
        mean = self.actor(obs)
        log_std = self.log_std
        z = (action - mean) * torch.exp(-log_std)
        logprob = -(0.5 * z * z + log_std + _LOG_SQRT_2PI).sum(-1)
        entropy = (log_std + _HALF_LOG_2PI_E).sum()
        return logprob, entropy, self.critic(obs).squeeze(-1)


class PPOTrainer:
    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        if cfg.device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.device = torch.device(cfg.device)
        self.envs = ThreadVecEnv(
            num_envs=cfg.num_envs,
            seed=cfg.seed,
            randomize=True,
            num_threads=cfg.num_threads,
        )
        self.net = ActorCritic(hidden=cfg.hidden, init_log_std=cfg.init_log_std).to(
            self.device
        )
        self.obs_rms = RunningMeanStd((C.OBS_DIM,)).to(self.device)
        self.ret_rms = RunningMeanStd(()).to(self.device)
        self.opt = torch.optim.Adam(
            self.net.parameters(), lr=cfg.lr, eps=1e-5, fused=(cfg.device == "cuda")
        )

        T, N = cfg.rollout_steps, cfg.num_envs
        dev = self.device
        self.buf_obs = torch.zeros((T, N, C.OBS_DIM), device=dev)
        self.buf_raw_obs = torch.zeros((T, N, C.OBS_DIM), device=dev)
        self.buf_actions = torch.zeros((T, N, C.ACT_DIM), device=dev)
        self.buf_logprobs = torch.zeros((T, N), device=dev)
        self.buf_rewards = torch.zeros((T, N), device=dev)
        self.buf_dones = torch.zeros((T, N), device=dev)
        self.buf_values = torch.zeros((T, N), device=dev)

        self.global_step = 0
        self.iteration = 0
        os.makedirs(cfg.out_dir, exist_ok=True)
        os.makedirs(cfg.video_dir, exist_ok=True)

    # --- rollout ---
    @torch.no_grad()
    def collect(self, obs):
        """obs is the raw observation tensor; the normalised copy the policy
        actually saw is what gets stored"""
        cfg, envs = self.cfg, self.envs
        for t in range(cfg.rollout_steps):
            norm_obs = self.obs_rms.normalize(obs)
            self.buf_raw_obs[t] = obs
            self.buf_obs[t] = norm_obs
            action, logprob, value = self.net.act(norm_obs)
            self.buf_actions[t] = action
            self.buf_logprobs[t] = logprob
            self.buf_values[t] = value

            np_action = action.clamp(-1.0, 1.0).cpu().numpy()
            next_obs, reward, term, trunc, final_obs = envs.step(np_action)

            r = torch.as_tensor(reward, device=self.device, dtype=torch.float32)
            term_t = torch.as_tensor(term, device=self.device)
            trunc_t = torch.as_tensor(trunc, device=self.device)

            if trunc.any():
                # Time-limit truncation is not a terminal state(this is important and I saw some implementtions missing this)
                fo = torch.as_tensor(final_obs, device=self.device, dtype=torch.float32)
                boot = self.net.value(self.obs_rms.normalize(fo))
                boot = self.ret_rms.denormalize(boot)
                r = r + cfg.gamma * boot * trunc_t.float()

            self.buf_rewards[t] = r
            self.buf_dones[t] = (term_t | trunc_t).float()
            obs = torch.as_tensor(next_obs, device=self.device, dtype=torch.float32)

        self.obs_rms.update(self.buf_raw_obs.reshape(-1, C.OBS_DIM))

        self.global_step += cfg.batch_size
        return obs

    @torch.no_grad()
    def compute_gae(self, last_obs):
        """Vectorised over envs"""
        cfg = self.cfg
        last_value = self.ret_rms.denormalize(
            self.net.value(self.obs_rms.normalize(last_obs))
        )
        values = self.ret_rms.denormalize(self.buf_values)

        advantages = torch.zeros_like(self.buf_rewards)
        last_gae = torch.zeros(cfg.num_envs, device=self.device)
        for t in reversed(range(cfg.rollout_steps)):
            next_non_terminal = 1.0 - self.buf_dones[t]
            next_value = last_value if t == cfg.rollout_steps - 1 else values[t + 1]
            delta = (
                self.buf_rewards[t]
                + cfg.gamma * next_value * next_non_terminal
                - values[t]
            )
            last_gae = delta + cfg.gamma * cfg.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + values
        return advantages, returns

    # --- update ---
    def update(self, advantages, returns):
        cfg = self.cfg
        self.ret_rms.update(returns.reshape(-1))
        target_values = (returns.reshape(-1) - self.ret_rms.mean) * torch.rsqrt(
            self.ret_rms.var + 1e-8
        )

        b_obs = self.buf_obs.reshape(-1, C.OBS_DIM)
        b_actions = self.buf_actions.reshape(-1, C.ACT_DIM)
        b_logprobs = self.buf_logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)

        stats = torch.zeros(5, device=self.device)  # pg, v, ent, kl, clipfrac
        n_updates = 0
        stop = False
        for epoch in range(cfg.num_epochs):
            perm = torch.randperm(cfg.batch_size, device=self.device)
            for start in range(0, cfg.batch_size, cfg.minibatch_size):
                idx = perm[start : start + cfg.minibatch_size]
                new_logprob, entropy, new_value = self.net.evaluate_actions(
                    b_obs[idx], b_actions[idx]
                )

                log_ratio = new_logprob - b_logprobs[idx]
                ratio = log_ratio.exp()

                mb_adv = b_advantages[idx]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef),
                ).mean()
                v_loss = 0.5 * (new_value - target_values[idx]).pow(2).mean()
                ent_loss = entropy.mean()
                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent_loss

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    # Schulman's low-variance KL estimator.
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clipfrac = ((ratio - 1.0).abs() > cfg.clip_coef).float().mean()
                    stats += torch.stack(
                        [
                            pg_loss.detach(),
                            v_loss.detach(),
                            ent_loss.detach(),
                            approx_kl,
                            clipfrac,
                        ]
                    )
                n_updates += 1

            if (
                cfg.target_kl is not None
                and (stats[3] / n_updates).item() > cfg.target_kl
            ):
                stop = True
                break
        return (stats / max(n_updates, 1)).tolist(), n_updates, stop

    # -- checkpointing ---
    def save(self, tag="latest"):
        path = os.path.join(self.cfg.out_dir, f"{self.cfg.run_name}_{tag}.pt")
        torch.save(
            {
                "net": self.net.state_dict(),
                "obs_rms": self.obs_rms.state_dict(),
                "ret_rms": self.ret_rms.state_dict(),
                "opt": self.opt.state_dict(),
                "cfg": asdict(self.cfg),
                "cmd_ranges": {
                    "vel": tuple(C.CMD_VEL_RANGE),
                    "yaw": tuple(C.CMD_YAW_RANGE),
                    "height": tuple(C.CMD_HEIGHT_RANGE),
                },
                "iteration": self.iteration,
                "global_step": self.global_step,
            },
            path,
        )
        return path

    def load(self, path, strict=True):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        ck = torch.load(path, map_location=self.device, weights_only=True)
        self.net.load_state_dict(ck["net"], strict=strict)
        self.obs_rms.load_state_dict(ck["obs_rms"])
        self.ret_rms.load_state_dict(ck["ret_rms"])
        if "opt" in ck:
            self.opt.load_state_dict(ck["opt"])
        self.iteration = ck.get("iteration", 0)
        self.global_step = ck.get("global_step", 0)
        return ck

    # -- main loop ---
    HEADER = (
        f"{'iter':>6} {'done':>6} {'steps':>12} {'sps':>7} {'eta':>6} │ "
        f"{'return':>8} {'len':>5} │ {'vel':>5} {'yaw':>5} {'hgt':>5} │ "
        f"{'kl':>7} {'std':>5}"
    )

    def train(self, resume=None):
        cfg = self.cfg
        if resume:
            self.load(resume)
            print(f"resumed from {resume} at iteration {self.iteration}")

        num_iterations = cfg.total_steps // cfg.batch_size
        obs = torch.as_tensor(
            self.envs.reset(), device=self.device, dtype=torch.float32
        )
        print(
            f"PPO on {self.device} | {cfg.num_envs} envs x {cfg.rollout_steps} steps "
            f"= {cfg.batch_size} batch | {num_iterations:,} iterations "
            f"| {cfg.total_steps:,} steps"
        )
        print(
            "  steps   total environment steps so far - the x-axis of any learning curve\n"
            "  sps     environment steps per second\n"
            "  return  reward summed over a whole episode; the analytic baseline scores 3400\n"
            "  len     steps survived before falling; 1000 means it never fell\n"
            "  vel/yaw/hgt\n"
            "          command tracking, 0 to 1 per step, 1 is exact. These are the\n"
            "          numbers to watch: return can rise on posture alone\n"
            "  kl      how far the update moved the policy; the run backs off above "
            f"{cfg.target_kl}\n"
            "  std     exploration noise, annealed by learning, not by schedule"
        )

        t_start = time.perf_counter()
        rows = 0

        self.start_iteration = self.iteration
        for it in range(self.iteration, num_iterations):
            self.iteration = it
            if cfg.anneal_lr:
                frac = 1.0 - it / num_iterations
                for g in self.opt.param_groups:
                    g["lr"] = frac * cfg.lr

            t0 = time.perf_counter()
            obs = self.collect(obs)
            advantages, returns = self.compute_gae(obs)
            (pg, vl, ent, kl, cf), n_up, stopped = self.update(advantages, returns)
            dt = time.perf_counter() - t0

            if it % cfg.log_every == 0:
                if rows % 25 == 0:
                    print(self.HEADER)
                rows += 1
                s = self.envs.pop_stats()
                std = self.net.log_std.exp().mean().item()
                done = (it + 1) / num_iterations
                eta = (
                    (num_iterations - it - 1)
                    * (time.perf_counter() - t_start)
                    / max(it - self.start_iteration + 1, 1)
                )
                tail = f"{kl:7.4f} {std:5.3f}"
                head = (
                    f"{it:6d} {100 * done:5.1f}% {self.global_step:12,d} "
                    f"{cfg.batch_size / dt:7,.0f} {_hms(eta):>6} │ "
                )
                if s:
                    print(
                        head + f"{s['return_mean']:8.1f} {s['length_mean']:5.0f} │ "
                        f"{s['r_vel']:5.3f} {s['r_yaw']:5.3f} {s['r_height']:5.3f} │ "
                        + tail
                        + ("  kl-stop" if stopped else "")
                    )
                else:
                    print(
                        head
                        + f"{'-':>8} {'-':>5} │ {'-':>5} {'-':>5} {'-':>5} │ "
                        + tail
                    )

            if cfg.record_every and it > 0 and it % cfg.record_every == 0:
                self.record(it)
            if cfg.save_every and it > 0 and it % cfg.save_every == 0:
                print(
                    f"       saved {self.save('latest')}  "
                    f"(iteration {it}, {self.global_step:,} steps)"
                )

        print(f"       saved {self.save('final')}")
        elapsed = time.perf_counter() - t_start
        print(
            f"done in {_hms(elapsed)} | "
            f"{self.global_step / elapsed:,.0f} env steps/s overall"
        )
        if cfg.record_every:
            self.record("final")
        return self

    # -- policy interface used by the recorder, evaluation and teleop --
    @torch.no_grad()
    def policy(self, obs_np, deterministic=True):
        obs = torch.as_tensor(np.asarray(obs_np, dtype=np.float32), device=self.device)
        single = obs.ndim == 1
        if single:
            obs = obs.unsqueeze(0)
        mean = self.net.actor(self.obs_rms.normalize(obs))
        if not deterministic:
            mean = mean + torch.randn_like(mean) * self.net.log_std.exp()
        out = mean.clamp(-1.0, 1.0).cpu().numpy()
        return out[0] if single else out

    def record(self, tag):
        """Record a fixed command schedule so clips are comparable across the run."""
        try:
            from recorder import record_policy

            path = record_policy(
                self.policy,
                os.path.join(
                    self.cfg.video_dir,
                    f"{self.cfg.run_name}_iter_{tag if isinstance(tag, str) else f'{tag:06d}'}.mp4",
                ),
                seconds=self.cfg.record_seconds,
                label=f"{self.cfg.run_name}  iter {tag}  step {self.global_step:,}",
            )
            print(f"       recorded {path}")
        except Exception as exc:  # never let recording kill a run
            print(f"       [recording skipped: {type(exc).__name__}: {exc}]")


def load_policy(path, device=None):
    """Load a checkpoint"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    ck = torch.load(path, map_location=device, weights_only=True)
    saved = ck.get("cfg", {})

    net = ActorCritic(
        hidden=tuple(saved.get("hidden", PPOConfig.hidden)),
        init_log_std=saved.get("init_log_std", PPOConfig.init_log_std),
    )
    net.load_state_dict(ck["net"])
    net.to(device).eval()
    obs_rms = RunningMeanStd((C.OBS_DIM,)).to(device)
    obs_rms.load_state_dict(ck["obs_rms"])
    ranges = ck.get("cmd_ranges") or {
        "vel": (-2.0, 2.0),
        "yaw": (-1.5, 1.5),
        "height": tuple(C.CMD_HEIGHT_RANGE),
    }
    ranges = {k: tuple(v) for k, v in ranges.items()}

    old_vel = 1.0 / ranges["vel"][1]
    old_yaw = 1.0 / ranges["yaw"][1]
    old_wheel = 1.0 / (3.0 * ranges["vel"][1])
    gain = np.ones(C.OBS_DIM, dtype=np.float32)
    gain[3:6] = old_vel / C.VEL_SCALE  # body linear velocity
    gain[17:19] = old_wheel / C.WHEEL_SCALE  # wheel angular rates
    gain[23] = old_vel / C.VEL_SCALE  # forward command
    gain[24] = old_yaw / C.YAW_SCALE  # yaw command
    rescale = None
    if not np.allclose(gain, 1.0):
        rescale = torch.as_tensor(gain, device=device)

    @torch.no_grad()
    def policy(obs_np):
        obs = torch.as_tensor(np.asarray(obs_np, dtype=np.float32), device=device)
        single = obs.ndim == 1
        if single:
            obs = obs.unsqueeze(0)
        if rescale is not None:
            obs = obs * rescale
        out = net.actor(obs_rms.normalize(obs)).clamp(-1.0, 1.0).cpu().numpy()
        return out[0] if single else out

    policy.iteration = ck.get("iteration", 0)
    policy.global_step = ck.get("global_step", 0)
    policy.cmd_ranges = ranges
    policy.stale = rescale is not None
    if policy.stale:
        print(
            f"note: {os.path.basename(path)} was trained with "
            f"CMD_VEL_RANGE={ranges['vel']}, config now says "
            f"{tuple(C.CMD_VEL_RANGE)}. It is capped at {ranges['vel'][1]:.1f} m/s "
            f"and its observations rescaled to match; retrain to drive faster."
        )
    return policy


def main():
    import argparse

    p = argparse.ArgumentParser(description="Train a velocity-tracking policy with PPO")
    p.add_argument("--steps", type=int, default=PPOConfig.total_steps)
    p.add_argument("--envs", type=int, default=PPOConfig.num_envs)
    p.add_argument("--rollout", type=int, default=PPOConfig.rollout_steps)
    p.add_argument("--lr", type=float, default=PPOConfig.lr)
    p.add_argument("--seed", type=int, default=PPOConfig.seed)
    p.add_argument("--name", type=str, default=PPOConfig.run_name)
    p.add_argument("--record-every", type=int, default=PPOConfig.record_every)
    p.add_argument(
        "--save-every",
        type=int,
        default=PPOConfig.save_every,
        help="checkpoint every N iterations (0 disables all but the final)",
    )
    p.add_argument("--log-every", type=int, default=PPOConfig.log_every)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--device", type=str, default=PPOConfig.device)
    a = p.parse_args()

    cfg = PPOConfig(
        total_steps=a.steps,
        num_envs=a.envs,
        rollout_steps=a.rollout,
        lr=a.lr,
        seed=a.seed,
        run_name=a.name,
        record_every=a.record_every,
        save_every=a.save_every,
        log_every=a.log_every,
        device=a.device,
    )
    PPOTrainer(cfg).train(resume=a.resume)


if __name__ == "__main__":
    main()
