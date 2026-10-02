"""Analytic cascade controller for the wheeled-legged robot.

This is the hand-tuned baseline the RL policies are measured against. It is a
classic two-wheeled-inverted-pendulum cascade
"""

import numpy as np
import robot_config as C


class BalanceController:
    def __init__(
        self,
        kp=14.0,
        kd=1.8,
        kv=0.40,
        ki=0.08,
        kyaw=0.40,
        kyaw_i=0.15,
        integral_limit=3.0,
        pitch_offset_limit=0.12,
        dt=C.CONTROL_DT,
    ):
        self.kp, self.kd = kp, kd  # inner pitch PD -> wheel torque
        self.kv, self.ki = kv, ki  # outer velocity PI -> pitch offset
        self.kyaw, self.kyaw_i = kyaw, kyaw_i  # yaw-rate PI -> torque differential
        self.integral_limit = integral_limit
        self.pitch_offset_limit = pitch_offset_limit
        self.dt = dt
        self.command = np.zeros(3)
        self._action = np.zeros(4)
        self.reset()

    def reset(self):
        self._integral = 0.0
        self._yaw_integral = 0.0
        self._action[:] = 0.0

    def set_command(self, v_cmd=None, yaw_cmd=None, height_cmd=None):
        """Set any subset of (forward velocity, yaw rate, ride height)."""
        if v_cmd is not None:
            self.command[0] = np.clip(v_cmd, *C.CMD_VEL_RANGE)
        if yaw_cmd is not None:
            self.command[1] = np.clip(yaw_cmd, *C.CMD_YAW_RANGE)
        if height_cmd is not None:
            self.command[2] = np.clip(height_cmd, *C.CMD_HEIGHT_RANGE)
        return self.command

    def act(self, data, command=None):
        """Return an action in [-1,1]^4, the same convention the RL env uses."""
        if command is not None:
            self.command[:] = command
        v_cmd, yaw_cmd, h_cmd = self.command

        _, pitch, pitch_rate, v_body, w_body = C.body_state(data)
        v_fwd = C.FORWARD_SIGN * v_body[1]
        yaw_rate = w_body[2]

        # Anti-windup by conditional integration
        err = v_cmd - v_fwd
        trial = self._integral + err * self.dt
        raw = self.kv * err + self.ki * trial
        if abs(raw) < self.pitch_offset_limit or raw * err < 0.0:
            self._integral = float(
                np.clip(trial, -self.integral_limit, self.integral_limit)
            )
        offset = np.clip(
            self.kv * err + self.ki * self._integral,
            -self.pitch_offset_limit,
            self.pitch_offset_limit,
        )
        pitch_des = C.PITCH_EQUILIBRIUM + offset

        # Inner loop: pitch PD. Not clipped here - the differential is added first and the sum is clipped
        u = self.kp * (pitch - pitch_des) + self.kd * pitch_rate

        yaw_err = yaw_cmd - yaw_rate
        self._yaw_integral = float(
            np.clip(
                self._yaw_integral + yaw_err * self.dt,
                -self.integral_limit,
                self.integral_limit,
            )
        )
        ud = C.YAW_DIFF_SIGN * (self.kyaw * yaw_err + self.kyaw_i * self._yaw_integral)

        self._action[0:2] = C.hip_to_action(C.height_to_hip(h_cmd))
        self._action[2] = np.clip(u + ud, -1.0, 1.0)
        self._action[3] = np.clip(u - ud, -1.0, 1.0)
        return self._action

    def apply(self, data, command=None):
        C.action_to_ctrl(self.act(data, command), out=data.ctrl)
        return data.ctrl


def rollout(
    model, data, command, steps=750, settle=0.55, controller=None, on_step=None
):
    """Run the controller from the stand keyframe"""
    import mujoco as mj

    ctl = controller or BalanceController()
    ctl.reset()
    schedule = [(0, *command)] if len(command) == 3 else sorted(command)
    ctl.set_command(*schedule[0][1:])
    nxt = 1

    mj.mj_resetDataKeyframe(model, data, 0)
    mj.mj_forward(model, data)

    acc = []
    for k in range(steps):
        while nxt < len(schedule) and k >= schedule[nxt][0]:
            ctl.set_command(*schedule[nxt][1:])
            nxt += 1
        ctl.apply(data)
        mj.mj_step(model, data, nstep=C.FRAME_SKIP)
        if on_step is not None:
            on_step(k, data)
        _, pitch, _, v_body, w_body = C.body_state(data)
        if abs(pitch) > 0.8 or data.qpos[2] < 0.6:
            return None, k * C.CONTROL_DT
        if k > steps * settle:
            acc.append((C.FORWARD_SIGN * v_body[1], w_body[2], data.qpos[2]))
    return np.array(acc).mean(0), steps * C.CONTROL_DT


if __name__ == "__main__":
    import mujoco as mj

    model = mj.MjModel.from_xml_path(C.XML_PATH)
    data = mj.MjData(model)

    print(
        "=== steady-state tracking (15 s per command, averaged over the last 6.7 s) ==="
    )
    print(f"{'v_cmd':>6} {'yaw_cmd':>8} {'h_cmd':>6} | {'v':>7} {'yaw':>7} {'z':>7}")
    for cmd in [
        (0.0, 0.0, 1.33),
        (1.0, 0.0, 1.33),
        (2.0, 0.0, 1.33),
        (-1.0, 0.0, 1.33),
        (-2.0, 0.0, 1.33),
        (0.0, 1.0, 1.33),
        (0.0, -1.5, 1.33),
        (1.0, 1.0, 1.33),
        (0.0, 0.0, 1.25),
        (0.0, 0.0, 1.15),
        (1.5, 0.8, 1.25),
        (-1.0, -1.0, 1.20),
        (2.0, 1.5, 1.15),
    ]:
        res, t = rollout(model, data, cmd)
        head = f"{cmd[0]:6.1f} {cmd[1]:8.1f} {cmd[2]:6.2f} |"
        print(
            f"{head} FELL @ {t:.1f}s"
            if res is None
            else f"{head} {res[0]:7.2f} {res[1]:7.2f} {res[2]:7.3f}"
        )

    print("\n=== step response (settling time to within 5%, peak overshoot) ===")
    print(f"{'step':>16} | {'t_settle':>9} {'overshoot':>10}")
    for label, cmd, idx in [
        ("v: 0 -> 1.0", (1.0, 0.0, 1.33), 0),
        ("v: 0 -> 2.0", (2.0, 0.0, 1.33), 0),
        ("v: 0 -> -2.0", (-2.0, 0.0, 1.33), 0),
        ("yaw: 0 -> 1.0", (0.0, 1.0, 1.33), 1),
        ("yaw: 0 -> -1.5", (0.0, -1.5, 1.33), 1),
    ]:
        trace = []
        rollout(
            model,
            data,
            cmd,
            steps=1000,
            on_step=lambda k, dd: trace.append(
                (C.FORWARD_SIGN * C.body_state(dd)[3][1], C.body_state(dd)[4][2])
            ),
        )
        sig = np.array(trace)[:, idx]
        target = cmd[idx]
        band = max(0.05 * abs(target), 0.03)
        bad = np.nonzero(np.abs(sig - target) > band)[0]
        ts = (
            (bad[-1] + 1) * C.CONTROL_DT
            if len(bad) and bad[-1] + 1 < len(sig)
            else np.inf
        )
        over = (sig.max() - target) if target > 0 else (target - sig.min())
        print(f"{label:>16} | {ts:8.1f}s {over:+10.3f}")

    print("\n=== teleop-style command schedule (mid-run changes) ===")
    sched = [
        (0, 1.0, 0.0, 1.33),
        (150, 2.0, 0.0, 1.33),
        (300, 2.0, 1.0, 1.20),
        (450, 0.0, 0.0, 1.20),
        (600, -1.5, -1.0, 1.33),
    ]
    res, t = rollout(model, data, sched, steps=750, settle=0.93)
    print(
        "  FELL @ %.1fs" % t
        if res is None
        else f"  survived {t:.0f}s, final v={res[0]:+.2f} yaw={res[1]:+.2f} z={res[2]:.3f} "
        f"(last cmd -1.5, -1.0, 1.33)"
    )
