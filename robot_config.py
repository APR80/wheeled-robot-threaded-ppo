"""Robot/task constants for the wheeled-legged robot."""

import numpy as np

XML_PATH = "robot_model/wheeled_robot.xml"

# --- Timing ---
PHYSICS_DT = 0.002
FRAME_SKIP = 10
CONTROL_DT = PHYSICS_DT * FRAME_SKIP
EPISODE_SECONDS = 20.0
MAX_EPISODE_STEPS = int(EPISODE_SECONDS / CONTROL_DT)  # 1000

# --- Geometry ---
WHEEL_RADIUS = 0.35
TRACK_WIDTH = 1.54  # wheel body x-positions +-0.77
FORWARD_SIGN = -1.0  # v_forward = FORWARD_SIGN * v_body[1]
YAW_DIFF_SIGN = -1.0
PITCH_EQUILIBRIUM = 0.0836  # rad (4.79 deg)

# --- Ride height ---
HEIGHT_MIN = 1.15
HEIGHT_MAX = 1.33
HEIGHT_TO_HIP_A = -1.577793
HEIGHT_TO_HIP_B = 2.095068
HIP_CTRL_MIN = 0.0  # <position ctrlrange> lower bound
HIP_CTRL_MAX = 0.30  # linkage hard stop

# --- Command ranges ---
CMD_VEL_RANGE = (-4.0, 4.0)
CMD_YAW_RANGE = (-1.5, 1.5)  # rad/s
CMD_HEIGHT_RANGE = (HEIGHT_MIN, HEIGHT_MAX)

# Observation scales
VEL_SCALE = 1.0 / CMD_VEL_RANGE[1]
YAW_SCALE = 1.0 / CMD_YAW_RANGE[1]
HEIGHT_MID = 0.5 * (HEIGHT_MIN + HEIGHT_MAX)
HEIGHT_SCALE = 1.0 / (0.5 * (HEIGHT_MAX - HEIGHT_MIN))
WHEEL_SCALE = 1.0 / (3.0 * CMD_VEL_RANGE[1])

# --- Termination ---
MAX_TILT = 0.6
MIN_HEIGHT = 0.95

# --- Joint indexing (qpos/qvel layout for the free-flyer + 8 hinges)
# qpos: [0:3] base xyz, [3:7] base quat(wxyz), then hinges in XML order.
# qvel: [0:3] base linear (WORLD frame), [3:6] base angular (BODY frame), then hinges.
QPOS_HIP = (7, 10)  # left_upperleg, right_upperleg
QPOS_KNEE = (8, 11)  # left_lowerleg, right_lowerleg
QPOS_WHEEL = (9, 12)  # left_wheel, right_wheel  (unbounded - never observe)
QVEL_HIP = (6, 9)
QVEL_KNEE = (7, 10)
QVEL_WHEEL = (8, 11)

OBS_DIM = 26
ACT_DIM = 4


HIP_CTRL_MID = 0.5 * (HIP_CTRL_MIN + HIP_CTRL_MAX)
HIP_CTRL_HALF = 0.5 * (HIP_CTRL_MAX - HIP_CTRL_MIN)


def height_to_hip(height):
    """Hip position command"""
    hip = HEIGHT_TO_HIP_A * np.asarray(height) + HEIGHT_TO_HIP_B
    return np.clip(hip, HIP_CTRL_MIN, HIP_CTRL_MAX)


def hip_to_height(hip):
    """Inverse of height_to_hip (no clipping - use to read back a hip command)."""
    return (np.asarray(hip) - HEIGHT_TO_HIP_B) / HEIGHT_TO_HIP_A


def action_to_ctrl(action, out=None):
    """Map a policy action in [-1,1]^4 to MuJoCo ctrl."""
    a = np.clip(action, -1.0, 1.0)
    out = np.empty(4) if out is None else out
    out[0:2] = HIP_CTRL_MID + HIP_CTRL_HALF * a[0:2]
    out[2:4] = a[2:4]
    return out


def hip_to_action(hip):
    """Inverse of the hip half of action_to_ctrl."""
    return (np.asarray(hip) - HIP_CTRL_MID) / HIP_CTRL_HALF


def quat_to_mat(quat):
    """Body to world rotation matrix from a MuJoCo (w,x,y,z) quaternion."""
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def body_state(data):
    """Extract the balance-relevant state from MjData."""
    R = quat_to_mat(data.qpos[3:7])
    g_body = R[2, :].copy()  # R.T @ [0,0,1]; down is -g_body
    pitch = np.arctan2(g_body[1], g_body[2])
    v_body = R.T @ data.qvel[0:3]  # qvel[0:3] is world-frame linear velocity
    w_body = data.qvel[3:6].copy()  # already body-frame
    return -g_body, pitch, w_body[0], v_body, w_body


if __name__ == "__main__":
    import mujoco as mj

    m = mj.MjModel.from_xml_path(XML_PATH)
    d = mj.MjData(m)
    mj.mj_resetDataKeyframe(m, d, 0)
    mj.mj_forward(m, d)
    mj.mj_comPos(m, d)
    com = (m.body_mass[:, None] * d.xipos).sum(0) / m.body_mass.sum()
    axle = float(
        d.xanchor[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "left_wheel_joint")][1]
    )
    print(f"total mass      {m.body_mass.sum():.1f} kg")
    print(f"COM             {np.round(com, 4)}   axle_y={axle:.3f}")
    print(
        f"equilibrium pitch {np.arctan2(com[1] - axle, com[2] - WHEEL_RADIUS):.4f} rad "
        f"(config says {PITCH_EQUILIBRIUM})"
    )
    print(
        f"obs dim {OBS_DIM}, act dim {ACT_DIM}, control {1 / CONTROL_DT:.0f} Hz, "
        f"episode {MAX_EPISODE_STEPS} steps"
    )
    for h in (HEIGHT_MIN, 0.5 * (HEIGHT_MIN + HEIGHT_MAX), HEIGHT_MAX):
        print(f"  height {h:.3f} m -> hip cmd {height_to_hip(h):.4f}")
