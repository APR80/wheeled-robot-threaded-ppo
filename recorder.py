"""High-quality offscreen video recording of a policy or controller.

Clips are meant to be cut together into a "training progress" video, so every clip
must be directly comparable: same camera, same lighting, same command schedule,
same duration. The only thing that differs between clips is the policy. That is why
`COMMAND_SCHEDULE` is a module constant rather than a parameter - change it and you
lose comparability with everything recorded before.

Rendering is EGL offscreen (no window, works headless) at 1920x1080. The model's
`<global offwidth/offheight>` must be at least this or `mujoco.Renderer` silently
caps at 640x480 and raises on the first frame. Frames are piped raw to ffmpeg
rather than written as PNGs: at 1080p50 that is the difference between a few
seconds and a few minutes per clip.
"""

import os
import shutil
import subprocess

# EGL must be selected before mujoco initialises its GL context.
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco as mj

import robot_config as C
from wheeled_legged_env import WheeledLeggedEnv

WIDTH, HEIGHT = 1920, 1080
FPS = int(round(1.0 / C.CONTROL_DT))  # 50, matching the control rate exactly

COMMAND_SCHEDULE = [
    (0.0, 0.0, 0.0, 1.33),  # stand still
    (1.5, 1.0, 0.0, 1.33),  # drive forward
    (3.5, 2.0, 0.0, 1.33),  # accelerate
    (5.5, 1.0, 1.2, 1.33),  # forward while turning
    (7.5, 0.0, -1.5, 1.20),  # spin in place, crouched
    (9.5, -1.5, 0.0, 1.20),  # reverse
    (11.0, 0.0, 0.0, 1.33),  # stop and stand tall
]


def command_at(t):
    cmd = COMMAND_SCHEDULE[0][1:]
    for entry in COMMAND_SCHEDULE:
        if t >= entry[0]:
            cmd = entry[1:]
    return np.array(cmd)


def _ffmpeg_path():
    path = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
    if not os.path.exists(path):
        raise FileNotFoundError("ffmpeg not found; install it to record videos")
    return path


def _font_path():
    """A font for the burned-in label. Returns None if none is found, in which
    case the label is simply dropped rather than failing the recording."""
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ):
        if os.path.exists(p):
            return p
    return None


def _escape_drawtext(text):
    """drawtext parses its own escapes, so ':' and '\\' have to be neutralised."""
    return text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "")


class VideoWriter:
    """Raw RGB frames in, H.264 out."""

    def __init__(self, path, width=WIDTH, height=HEIGHT, fps=FPS, crf=18, label=None):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.path = path
        cmd = [
            _ffmpeg_path(),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
        ]
        font = _font_path() if label else None
        if font:
            size = max(20, height // 30)
            cmd += [
                "-vf",
                (
                    f"drawtext=fontfile={font}:text='{_escape_drawtext(label)}'"
                    f":fontcolor=white:fontsize={size}:x={size}:y={size}"
                    f":box=1:boxcolor=black@0.45:boxborderw={size // 3}"
                ),
            ]
        cmd += [
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            path,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def add(self, frame):
        self.proc.stdin.write(frame.tobytes())

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()
        return self.path

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def record_policy(
    policy, path, seconds=12.0, camera="track", seed=0, label=None, randomize=False
):
    """Render `policy` following COMMAND_SCHEDULE and write an mp4."""
    env = WheeledLeggedEnv(
        seed=seed,
        randomize=randomize,
        max_steps=int(seconds / C.CONTROL_DT) + 10,
        command_resample_seconds=1e6,
    )
    obs = env.reset(command=command_at(0.0))

    baseline = None
    if policy is None:
        from baseline_controller import BalanceController

        baseline = BalanceController()
        baseline.reset()

    renderer = mj.Renderer(env.model, height=HEIGHT, width=WIDTH)
    scene_option = mj.MjvOption()
    scene_option.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = False

    steps = int(seconds / C.CONTROL_DT)
    writer = VideoWriter(path, label=label)
    try:
        for k in range(steps):
            t = k * C.CONTROL_DT
            env.set_command(command_at(t))

            if baseline is not None:
                action = baseline.act(env.data, env.command)
            else:
                action = policy(obs)

            renderer.update_scene(env.data, camera=camera, scene_option=scene_option)
            writer.add(renderer.render())

            obs, _, term, trunc, _ = env.step(action)
            if term:
                # Hold the last frame briefly so a fall reads as a fall on screen
                # rather than as an abrupt cut.
                renderer.update_scene(
                    env.data, camera=camera, scene_option=scene_option
                )
                frame = renderer.render()
                for _ in range(FPS // 2):
                    writer.add(frame)
                break
    finally:
        writer.close()
        renderer.close()
    return path


def record_baseline(path="videos/baseline_controller.mp4", **kwargs):
    return record_policy(None, path, **kwargs)


def record_checkpoint(checkpoint, path=None, **kwargs):
    """Record a saved PPO checkpoint."""
    import torch
    from vect_ppo import PPOTrainer, PPOConfig

    ck = torch.load(checkpoint, map_location="cpu")
    cfg = PPOConfig(
        **{k: v for k, v in ck["cfg"].items() if k in PPOConfig.__dataclass_fields__}
    )
    cfg.num_envs, cfg.num_threads, cfg.record_every = 1, 1, 0
    trainer = PPOTrainer(cfg)
    trainer.load(checkpoint)
    if path is None:
        base = os.path.splitext(os.path.basename(checkpoint))[0]
        path = os.path.join(cfg.video_dir, f"{base}.mp4")
    out = record_policy(trainer.policy, path, **kwargs)
    trainer.envs.close()
    return out


if __name__ == "__main__":
    import argparse
    import time

    p = argparse.ArgumentParser(
        description="Record a policy or the baseline controller"
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="PPO checkpoint to record; omit to record the baseline",
    )
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--camera", type=str, default="track", choices=["track", "side"])
    a = p.parse_args()

    t0 = time.perf_counter()
    if a.checkpoint:
        out = record_checkpoint(a.checkpoint, a.out, seconds=a.seconds, camera=a.camera)
    else:
        out = record_baseline(
            a.out or "videos/baseline_controller.mp4",
            seconds=a.seconds,
            camera=a.camera,
        )
    dt = time.perf_counter() - t0
    size = os.path.getsize(out) / 1e6
    print(
        f"wrote {out}  ({size:.1f} MB, {a.seconds:.0f}s at {WIDTH}x{HEIGHT}{FPS}, "
        f"rendered in {dt:.1f}s)"
    )
