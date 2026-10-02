"""Threaded vectorised environment."""

import os
import copy
import numpy as np
from concurrent.futures import ThreadPoolExecutor

import mujoco as mj

import robot_config as C
import wheeled_legged_env as E


class ThreadVecEnv:
    """Synchronous vectorised env with automatic reset on episode end."""

    def __init__(
        self,
        num_envs=32,
        seed=0,
        randomize=True,
        num_threads=None,
        xml_path=C.XML_PATH,
        max_steps=C.MAX_EPISODE_STEPS,
        command_resample_seconds=5.0,
        push_prob=0.002,
        share_model=False,
    ):
        self.num_envs = num_envs
        self.randomize = randomize
        self.max_steps = max_steps
        self.resample_every = int(command_resample_seconds / C.CONTROL_DT)
        self.push_prob = push_prob
        self.rng = np.random.default_rng(seed)

        base = mj.MjModel.from_xml_path(xml_path)
        self.model = base
        self._nominal_mass = base.body_mass.copy()
        self._nominal_friction = base.geom_friction.copy()

        if num_threads is None:
            num_threads = min(num_envs, os.cpu_count() or 4)
        self.num_threads = max(1, min(num_threads, num_envs))
        bounds = np.linspace(0, num_envs, self.num_threads + 1).astype(int)
        self._slices = [
            range(bounds[i], bounds[i + 1])
            for i in range(self.num_threads)
            if bounds[i] < bounds[i + 1]
        ]
        self._pool = ThreadPoolExecutor(max_workers=self.num_threads)

        self._per_thread_model = randomize and not share_model
        n_models = len(self._slices) if self._per_thread_model else 1
        self._models = [base] + [copy.deepcopy(base) for _ in range(n_models - 1)]
        self.datas = [mj.MjData(base) for _ in range(num_envs)]
        self._env_mass = np.tile(self._nominal_mass, (num_envs, 1))
        self._env_fric = np.ones(num_envs)

        # env index -> index of the model its thread owns.
        self._model_of = np.zeros(num_envs, dtype=np.int64)
        if self._per_thread_model:
            for k, sl in enumerate(self._slices):
                self._model_of[sl.start : sl.stop] = k

        nq, nv = base.nq, base.nv
        self.qpos = np.zeros((num_envs, nq))
        self.qvel = np.zeros((num_envs, nv))
        self.obs = np.zeros((num_envs, C.OBS_DIM), dtype=np.float32)
        self.final_obs = np.zeros((num_envs, C.OBS_DIM), dtype=np.float32)
        self.rewards = np.zeros(num_envs, dtype=np.float32)
        self.terminated = np.zeros(num_envs, dtype=bool)
        self.truncated = np.zeros(num_envs, dtype=bool)
        self.commands = np.zeros((num_envs, 3))

        # Scratch for command resampling.
        self._cmd_scratch = np.zeros((num_envs, 3))
        self.prev_action = np.zeros((num_envs, C.ACT_DIM))
        self._action = np.zeros((num_envs, C.ACT_DIM))
        self._ctrl = np.zeros((num_envs, C.ACT_DIM))
        self.step_counts = np.zeros(num_envs, dtype=np.int64)
        self.returns = np.zeros(num_envs)

        self.episode_returns = []
        self.episode_lengths = []
        self.episode_tracking = []
        # Tracking rewards accumulated over the episode.
        self._track_sum = np.zeros((num_envs, 3))

    # -- threaded region: physics only ---
    def _physics_slice(self, k):
        """Step every env in thread `k`'s slice, using thread `k`'s own model."""
        ctrl, qpos, qvel = self._ctrl, self.qpos, self.qvel
        datas, fs = self.datas, C.FRAME_SKIP
        m = self._models[k if self._per_thread_model else 0]
        if self._per_thread_model:
            mass, fric = self._env_mass, self._env_fric
            nom_fric = self._nominal_friction[:, 0]
            for i in self._slices[k]:
                d = datas[i]
                m.body_mass[:] = mass[i]
                m.geom_friction[:, 0] = nom_fric * fric[i]
                d.ctrl[:] = ctrl[i]
                mj.mj_step(m, d, nstep=fs)
                qpos[i] = d.qpos
                qvel[i] = d.qvel
        else:
            for i in self._slices[k]:
                d = datas[i]
                d.ctrl[:] = ctrl[i]
                mj.mj_step(m, d, nstep=fs)
                qpos[i] = d.qpos
                qvel[i] = d.qvel

    def _run_physics(self):
        if len(self._slices) == 1:
            self._physics_slice(0)
        else:
            list(self._pool.map(self._physics_slice, range(len(self._slices))))

    # -- main-thread batch work --
    def _observe_all(self):
        g, v, w = E.batch_body_state(self.qpos[:, 3:7], self.qvel)
        self._gravity, self._v_body, self._w_body = g, v, w
        E.batch_observe(
            self.qpos, self.qvel, self.prev_action, self.commands, g, v, w, self.obs
        )
        return self.obs

    def _reset_indices(self, idx):
        """Reset the envs named by idx and refresh their rows."""
        if self.randomize:
            if self._per_thread_model:
                mass, fric = E.sample_dynamics(self.rng, self._nominal_mass, len(idx))
                self._env_mass[idx] = mass
                self._env_fric[idx] = fric
            else:
                E.randomize_dynamics(
                    self._models[0],
                    self._nominal_mass,
                    self._nominal_friction,
                    self.rng,
                )
        # Resets run on the main thread with every worker idle
        for i in idx:
            m = self._models[self._model_of[i]]
            if self._per_thread_model:
                m.body_mass[:] = self._env_mass[i]
                m.geom_friction[:, 0] = self._nominal_friction[:, 0] * self._env_fric[i]
            E.reset_state(m, self.datas[i], self.rng, self.randomize)
            self.qpos[i] = self.datas[i].qpos
            self.qvel[i] = self.datas[i].qvel
        self.prev_action[idx] = 0.0
        self.step_counts[idx] = 0
        self.returns[idx] = 0.0
        self._track_sum[idx] = 0.0
        self._resample_commands(idx)

    def _resample_commands(self, idx):
        """Draw fresh commands for `idx`. See `_cmd_scratch` on why this is not a
        one-liner."""
        buf = self._cmd_scratch[: len(idx)]
        E.sample_command(self.rng, buf)
        self.commands[idx] = buf

    # -- API ---
    def reset(self):
        self._reset_indices(np.arange(self.num_envs))
        return self._observe_all()

    def step(self, actions):
        np.copyto(self._action, actions, casting="unsafe")
        E.batch_action_to_ctrl(self._action, self._ctrl)

        if self.randomize and self.push_prob > 0.0:
            # Random shoves
            pushed = np.nonzero(self.rng.random(self.num_envs) < self.push_prob)[0]
            for i in pushed:
                self.datas[i].qvel[0:2] += self.rng.normal(0.0, 0.35, size=2)

        self._run_physics()
        self.step_counts += 1
        self._observe_all()

        r, r_vel, r_yaw, r_height = E.batch_reward(
            self.qpos,
            self.qvel,
            self.commands,
            self._action,
            self.prev_action,
            self._gravity,
            self._v_body,
            self._w_body,
        )
        self.prev_action[:] = self._action
        self.returns += r
        self._track_sum[:, 0] += r_vel
        self._track_sum[:, 1] += r_yaw
        self._track_sum[:, 2] += r_height

        term = E.batch_terminated(self.qpos, self._gravity)
        bad = ~np.isfinite(self.obs).all(
            1
        )  # a diverged solver must not poison the buffer
        term |= bad
        trunc = self.step_counts >= self.max_steps
        r = r - E.FALL_PENALTY * term

        self.rewards[:] = r
        self.terminated[:] = term
        self.truncated[:] = trunc

        done = term | trunc
        if done.any():
            idx = np.nonzero(done)[0]
            self.final_obs[idx] = self.obs[idx]
            self.episode_returns.extend(self.returns[idx].tolist())
            self.episode_lengths.extend(self.step_counts[idx].tolist())
            self.episode_tracking.extend(
                (self._track_sum[idx] / self.step_counts[idx, None]).tolist()
            )
            self._reset_indices(idx)
            # Recompute only the reset rows, not the whole batch.
            g, v, w = E.batch_body_state(self.qpos[idx, 3:7], self.qvel[idx])
            sub = np.zeros((len(idx), C.OBS_DIM), dtype=np.float32)
            E.batch_observe(
                self.qpos[idx],
                self.qvel[idx],
                self.prev_action[idx],
                self.commands[idx],
                g,
                v,
                w,
                sub,
            )
            self.obs[idx] = sub

        due = np.nonzero(
            (self.step_counts % self.resample_every == 0)
            & (self.step_counts > 0)
            & ~done
        )[0]
        if len(due):
            self._resample_commands(due)

        return self.obs, self.rewards, self.terminated, self.truncated, self.final_obs

    def set_commands(self, command):
        """Pin every env to the same command (used for evaluation and recording)."""
        self.commands[:] = command

    def pop_stats(self):
        """Drain and summarise episodes completed since the last call."""
        if not self.episode_returns:
            return None
        track = np.array(self.episode_tracking).mean(0)
        stats = {
            "episodes": len(self.episode_returns),
            "return_mean": float(np.mean(self.episode_returns)),
            "return_std": float(np.std(self.episode_returns)),
            "length_mean": float(np.mean(self.episode_lengths)),
            "r_vel": float(track[0]),
            "r_yaw": float(track[1]),
            "r_height": float(track[2]),
        }
        self.episode_returns.clear()
        self.episode_lengths.clear()
        self.episode_tracking.clear()
        return stats

    def close(self):
        self._pool.shutdown(wait=True)

    def __del__(self):
        try:
            self._pool.shutdown(wait=False)
        except Exception:
            pass


if __name__ == "__main__":
    import time
    import resource

    def rss_mb():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    print(
        f"{'envs':>5} {'threads':>8} {'steps/s':>12} {'us/env-step':>12} "
        f"{'vs 1-thread':>12} {'RSS MB':>8}"
    )
    ref = None
    for num_envs, threads in [
        (1, 1),
        (16, 1),
        (16, 6),
        (16, 12),
        (48, 12),
        (128, 12),
        (256, 12),
        (512, 12),
    ]:
        venv = ThreadVecEnv(num_envs=num_envs, num_threads=threads, seed=0)
        venv.reset()
        a = np.zeros((num_envs, C.ACT_DIM), dtype=np.float32)
        for _ in range(20):
            venv.step(a)
        iters = max(50, int(6000 / num_envs))
        t0 = time.perf_counter()
        for _ in range(iters):
            venv.step(a)
        dt = time.perf_counter() - t0
        sps = num_envs * iters / dt
        if ref is None:
            ref = sps
        print(
            f"{num_envs:5d} {threads:8d} {sps:12,.0f} {dt / (iters * num_envs) * 1e6:12.1f} "
            f"{sps / ref:11.1f}x {rss_mb():8.0f}"
        )
        venv.close()
