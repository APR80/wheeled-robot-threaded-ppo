"""Command-conditioned velocity-tracking environment for the wheeled-legged robot."""

import numpy as np
import mujoco as mj

import robot_config as C


W_VEL, W_YAW, W_HEIGHT, W_UPRIGHT = 1.5, 1.0, 0.8, 0.5
P_LATERAL, P_ROLL, P_ACTION, P_TORQUE = 0.05, 0.02, 0.05, 0.01
P_VERT = 0.5
FALL_PENALTY = 5.0
HEIGHT_SIGMA = 0.13


# Batch kernels
def batch_body_state(quat, qvel):
    """Body-frame state for N robots."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    # Columns of the body->world rotation matrix.
    c0 = np.stack([1 - 2 * (yy + zz), 2 * (xy + wz), 2 * (xz - wy)], axis=1)
    c1 = np.stack([2 * (xy - wz), 1 - 2 * (xx + zz), 2 * (yz + wx)], axis=1)
    c2 = np.stack([2 * (xz + wy), 2 * (yz - wx), 1 - 2 * (xx + yy)], axis=1)

    v_world = qvel[:, 0:3]
    v_body = np.stack(
        [(c0 * v_world).sum(1), (c1 * v_world).sum(1), (c2 * v_world).sum(1)], axis=1
    )
    # gravity in the body frame = R^T @ (0,0,-1) = -(third ROW of R).
    gravity = -np.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], axis=1)
    w_body = qvel[:, 3:6]
    return gravity, v_body, w_body


def batch_observe(qpos, qvel, prev_action, command, gravity, v_body, w_body, out):
    """Fill `out` (N,26) with the policy observation."""
    out[:, 0:3] = gravity
    out[:, 3:6] = v_body * C.VEL_SCALE
    out[:, 6:9] = w_body * 0.25
    out[:, 9] = qpos[:, C.QPOS_HIP[0]] * 3.0  # hip range is only 0..0.3 rad
    out[:, 10] = qpos[:, C.QPOS_HIP[1]] * 3.0
    out[:, 11] = qpos[:, C.QPOS_KNEE[0]]
    out[:, 12] = qpos[:, C.QPOS_KNEE[1]]
    out[:, 13] = qvel[:, C.QVEL_HIP[0]] * 0.25
    out[:, 14] = qvel[:, C.QVEL_HIP[1]] * 0.25
    out[:, 15] = qvel[:, C.QVEL_KNEE[0]] * 0.25
    out[:, 16] = qvel[:, C.QVEL_KNEE[1]] * 0.25
    out[:, 17] = qvel[:, C.QVEL_WHEEL[0]] * C.WHEEL_SCALE
    out[:, 18] = qvel[:, C.QVEL_WHEEL[1]] * C.WHEEL_SCALE
    out[:, 19:23] = prev_action
    out[:, 23] = command[:, 0] * C.VEL_SCALE
    out[:, 24] = command[:, 1] * C.YAW_SCALE
    out[:, 25] = (command[:, 2] - C.HEIGHT_MID) * C.HEIGHT_SCALE
    return out


def batch_reward(qpos, qvel, command, action, prev_action, gravity, v_body, w_body):
    """Returns (reward, r_vel, r_yaw, r_height), each (N,)."""
    v_fwd = C.FORWARD_SIGN * v_body[:, 1]
    v_err = command[:, 0] - v_fwd
    yaw_err = command[:, 1] - w_body[:, 2]
    h_err = (command[:, 2] - qpos[:, 2]) / HEIGHT_SIGMA

    # Exponential tracking terms
    r_vel = np.exp(-4.0 * v_err * v_err)
    r_yaw = np.exp(-4.0 * yaw_err * yaw_err)
    r_height = np.exp(-h_err * h_err)
    upright = -gravity[:, 2]  # +1 when perfectly upright

    p_lateral = v_body[:, 0] ** 2
    p_roll = w_body[:, 1] ** 2
    p_action = ((action - prev_action) ** 2).sum(1)
    p_torque = (action[:, 2:4] ** 2).sum(1)
    p_vert = qvel[:, 2] ** 2  # world-frame vertical speed

    reward = (
        W_VEL * r_vel
        + W_YAW * r_yaw
        + W_HEIGHT * r_height
        + W_UPRIGHT * upright
        - P_LATERAL * p_lateral
        - P_ROLL * p_roll
        - P_ACTION * p_action
        - P_TORQUE * p_torque
        - P_VERT * p_vert
    )
    return reward, r_vel, r_yaw, r_height


def batch_terminated(qpos, gravity):
    """Fallen: tilted past MAX_TILT or sunk below MIN_HEIGHT."""
    tilt = np.arccos(np.clip(-gravity[:, 2], -1.0, 1.0))
    return (tilt > C.MAX_TILT) | (qpos[:, 2] < C.MIN_HEIGHT)


def batch_action_to_ctrl(action, out):
    """(N,4) policy action in [-1,1] -> (N,4) MuJoCo ctrl."""
    np.clip(action, -1.0, 1.0, out=out)
    out[:, 0:2] *= C.HIP_CTRL_HALF
    out[:, 0:2] += C.HIP_CTRL_MID
    return out


# Shared episode logic (used by both the single env and the vectorised one)
def sample_command(rng, out, n=None):
    """Draw commands into `out` (N,3)."""
    n = out.shape[0] if n is None else n
    r = rng.random(n)
    out[:, 0] = rng.uniform(*C.CMD_VEL_RANGE, size=n)
    out[:, 1] = rng.uniform(*C.CMD_YAW_RANGE, size=n)
    out[:, 2] = rng.uniform(*C.CMD_HEIGHT_RANGE, size=n)
    stop = r < 0.15
    spin = (r >= 0.15) & (r < 0.25)
    out[stop, 0] = 0.0
    out[stop, 1] = 0.0
    out[spin, 0] = 0.0
    return out


def sample_dynamics(rng, nominal_mass, n):
    """Draw `n` randomised dynamics parameter sets around the nominal model."""
    scale = rng.uniform(0.92, 1.08, size=(n, nominal_mass.shape[0]))
    scale[:, 1] = rng.uniform(0.85, 1.15, size=n)  # base body: +-15% payload
    return nominal_mass * scale, rng.uniform(0.7, 1.3, size=n)


def randomize_dynamics(model, nominal_mass, nominal_friction, rng):
    """Perturb masses and ground friction around the nominal model."""
    scale = rng.uniform(0.92, 1.08, size=nominal_mass.shape)
    scale[1] = rng.uniform(0.85, 1.15)  # base body: +-15% payload
    model.body_mass[:] = nominal_mass * scale
    model.geom_friction[:, 0] = nominal_friction[:, 0] * rng.uniform(0.7, 1.3)


def reset_state(model, data, rng, randomize):
    """Reset to the stand keyframe, optionally with start-state noise."""
    mj.mj_resetDataKeyframe(model, data, 0)
    if randomize:
        data.qpos[2] += rng.uniform(-0.02, 0.02)
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        mj.mju_axisAngle2Quat(data.qpos[3:7], axis, rng.uniform(-0.08, 0.08))
        data.qpos[7:] += rng.uniform(-0.02, 0.02, size=model.nq - 7)
        data.qvel[:] = rng.uniform(-0.05, 0.05, size=model.nv)
        data.qvel[3:6] *= 2.0
    mj.mj_forward(model, data)


# Single environment
class WheeledLeggedEnv:
    """One MuJoCo instance, Gym-like. Used by teleop, the recorder and evaluation;
    ThreadVecEnv is what training runs on."""

    observation_space_shape = (C.OBS_DIM,)
    action_space_shape = (C.ACT_DIM,)

    def __init__(
        self,
        xml_path=C.XML_PATH,
        seed=None,
        randomize=True,
        max_steps=C.MAX_EPISODE_STEPS,
        command_resample_seconds=5.0,
        model=None,
        push_prob=0.002,
    ):
        self.model = model if model is not None else mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.randomize = randomize
        self.max_steps = max_steps
        self.resample_every = int(command_resample_seconds / C.CONTROL_DT)
        self.push_prob = push_prob

        self._nominal_mass = self.model.body_mass.copy()
        self._nominal_friction = self.model.geom_friction.copy()

        # 1-row buffers so the batch kernels can be used unchanged.
        self._obs = np.zeros((1, C.OBS_DIM), dtype=np.float32)
        self._qpos = np.zeros((1, self.model.nq))
        self._qvel = np.zeros((1, self.model.nv))
        self._ctrl = np.zeros((1, 4))
        self._action = np.zeros((1, 4))
        self.prev_action = np.zeros((1, 4))
        self._command = np.zeros((1, 3))
        self.step_count = 0
        self.episode_return = 0.0

    @property
    def command(self):
        return self._command[0]

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def sample_command(self):
        sample_command(self.rng, self._command)
        return self.command

    def set_command(self, command):
        self._command[0] = command

    def _sync(self):
        self._qpos[0] = self.data.qpos
        self._qvel[0] = self.data.qvel
        g, v, w = batch_body_state(self._qpos[:, 3:7], self._qvel)
        self._gravity, self._v_body, self._w_body = g, v, w
        batch_observe(
            self._qpos, self._qvel, self.prev_action, self._command, g, v, w, self._obs
        )
        return self._obs[0]

    def reset(self, command=None):
        if self.randomize:
            randomize_dynamics(
                self.model, self._nominal_mass, self._nominal_friction, self.rng
            )
        reset_state(self.model, self.data, self.rng, self.randomize)
        self.prev_action[:] = 0.0
        self.step_count = 0
        self.episode_return = 0.0
        if command is None:
            self.sample_command()
        else:
            self.set_command(command)
        return self._sync()

    def step(self, action):
        self._action[0] = action
        batch_action_to_ctrl(self._action, self._ctrl)
        self.data.ctrl[:] = self._ctrl[0]

        if self.randomize and self.rng.random() < self.push_prob:
            # Random shove
            self.data.qvel[0:2] += self.rng.normal(0.0, 0.35, size=2)

        mj.mj_step(self.model, self.data, nstep=C.FRAME_SKIP)
        self.step_count += 1

        obs = self._sync()
        r, r_vel, r_yaw, r_height = batch_reward(
            self._qpos,
            self._qvel,
            self._command,
            self._action,
            self.prev_action,
            self._gravity,
            self._v_body,
            self._w_body,
        )
        reward = float(r[0])
        self.prev_action[:] = self._action
        self.episode_return += reward

        terminated = bool(
            batch_terminated(self._qpos, self._gravity)[0] or not np.isfinite(obs).all()
        )
        truncated = self.step_count >= self.max_steps
        if terminated:
            reward -= FALL_PENALTY

        if self.step_count % self.resample_every == 0 and not (terminated or truncated):
            self.sample_command()

        info = {
            "r_vel": float(r_vel[0]),
            "r_yaw": float(r_yaw[0]),
            "r_height": float(r_height[0]),
            "episode_return": self.episode_return,
            "length": self.step_count,
        }
        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    import time
    from baseline_controller import BalanceController

    env = WheeledLeggedEnv(seed=0, randomize=True)
    obs = env.reset()
    print(f"obs {obs.shape} dtype {obs.dtype}  finite={np.isfinite(obs).all()}")
    print(f"obs range [{obs.min():+.2f}, {obs.max():+.2f}]")

    ctl = BalanceController()
    returns, lengths, tracking = [], [], []
    for ep in range(10):
        env.reset()
        ctl.reset()
        done = False
        while not done:
            ctl.set_command(*env.command)
            obs, r, term, trunc, info = env.step(ctl.act(env.data))
            tracking.append((info["r_vel"], info["r_yaw"], info["r_height"]))
            done = term or trunc
        returns.append(info["episode_return"])
        lengths.append(info["length"])
    tr = np.array(tracking).mean(0)
    print("\nbaseline controller over 10 episodes:")
    print(f"  return {np.mean(returns):8.1f} +- {np.std(returns):.1f}")
    print(f"  length {np.mean(lengths):8.1f} / {C.MAX_EPISODE_STEPS}")
    print(f"  mean r_vel {tr[0]:.3f}  r_yaw {tr[1]:.3f}  r_height {tr[2]:.3f}")

    env.randomize = False
    env.reset()
    a = np.zeros(4)
    t0 = time.perf_counter()
    N = 2000
    for _ in range(N):
        _, _, term, trunc, _ = env.step(a)
        if term or trunc:
            env.reset()
    dt = time.perf_counter() - t0
    print(
        f"\nthroughput (single env, zero action): {N / dt:,.0f} steps/s "
        f"({dt / N * 1e6:.0f} us/step)"
    )
