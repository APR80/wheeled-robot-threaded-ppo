"""Drive the wheeled-legged robot from the keyboard.

W / S       drive forward / back      (hold)
A / D       turn left / right         (hold)
R / F       raise / lower ride height (hold)
SPACE       stop: zero velocity and yaw
TAB         next controller (trained policy <-> analytic baseline)
P           shove the robot, to watch it recover
BACKSPACE   reset the episode
F1/F2/F3    toggle the readout / key caps / key list
F4          toggle burning the overlay into the recording
F9          start or stop recording
ESC / Q     quit
"""

import argparse
import glob
import os
import shutil
import subprocess
import time

import numpy as np
import glfw
import mujoco as mj
from PIL import Image

import robot_config as C
from wheeled_legged_env import WheeledLeggedEnv

VEL_RATE, YAW_RATE, HEIGHT_RATE = 3.0, 2.0, 0.12
VEL_DECAY, YAW_DECAY = 3.5, 3.0
SHOVE_SIGMA = 0.6
WINDOW_W, WINDOW_H = 1920, 1080  # default window; the video size is separate

REC_W, REC_H = 1920, 1080  # video size, independent of the window
REC_FPS = 50
REC_CRF = 18  # visually lossless enough for a progress video
REC_PRESET = "slow"
REC_SAMPLES = 4  # MSAA for the offscreen recording buffer


def _clamp(x, bounds):
    lo, hi = bounds
    return float(min(max(x, lo), hi))


def _toward_zero(x, step):
    return float(np.sign(x) * max(abs(x) - step, 0.0))


class VideoWriter:
    """Raw RGB frames in, H.264 out, through an ffmpeg pipe."""

    def __init__(self, path, w, h, fps=REC_FPS, crf=REC_CRF, preset=REC_PRESET):
        if w < 2 or h < 2 or w % 2 or h % 2:
            raise RuntimeError("video width and height must be positive even integers")
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is not on PATH - cannot record")
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.path, self.w, self.h, self.frames = path, w, h, 0
        self.proc = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{w}x{h}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-vf",
                "vflip",
                "-c:v",
                "libx264",
                "-preset",
                preset,
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                path,
            ],
            stdin=subprocess.PIPE,
        )

    def add(self, rgb):
        try:
            self.proc.stdin.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
            self.frames += 1
        except (BrokenPipeError, ValueError):
            pass  # ffmpeg died

    def close(self):
        try:
            self.proc.stdin.close()
        except (BrokenPipeError, ValueError):
            pass
        self.proc.wait()
        return self.path, self.frames


class BaselineDriver:
    """The analytic controller"""

    name = "analytic baseline"
    max_speed = 6.0
    max_yaw = C.CMD_YAW_RANGE[1]

    def __init__(self, lean_limit=0.30):
        from baseline_controller import BalanceController

        self.ctl = BalanceController(pitch_offset_limit=lean_limit)

    def reset(self):
        self.ctl.reset()

    def act(self, obs, command, data):
        self.ctl.set_command(*command)
        return self.ctl.act(data)


class PolicyDriver:
    """A trained checkpoint"""

    def __init__(self, fn, name):
        self.fn = fn
        self.name = name
        self.max_speed = fn.cmd_ranges["vel"][1]
        self.max_yaw = fn.cmd_ranges["yaw"][1]

    def reset(self):
        pass

    def act(self, obs, command, data):
        return self.fn(obs)


KEYCAPS = (
    (0, 1, 1, "W", (glfw.KEY_W,), False),
    (0, 3, 1, "R", (glfw.KEY_R,), False),
    (1, 0, 1, "A", (glfw.KEY_A,), False),
    (1, 1, 1, "S", (glfw.KEY_S,), False),
    (1, 2, 1, "D", (glfw.KEY_D,), False),
    (1, 3, 1, "F", (glfw.KEY_F,), False),
    (2, 0, 3, "SPACE", (glfw.KEY_SPACE,), True),
    (2, 3, 1, "TAB", (glfw.KEY_TAB,), True),
    (3, 0, 2, "BKSP", (glfw.KEY_BACKSPACE,), True),
    (3, 2, 1, "P", (glfw.KEY_P,), True),
)
CAP_HELD = (0.0, 0.78, 1.0)
CAP_SHOT = (1.0, 0.28, 0.18)
CAP_IDLE = (0.16, 0.18, 0.21)


def newest_checkpoint(out_dir="runs"):
    """Most recently modified checkpoint, or None"""
    paths = glob.glob(os.path.join(out_dir, "*.pt"))
    if not paths:
        return None
    return max(paths, key=lambda p: (os.path.getmtime(p), p.endswith("_final.pt")))


class Teleop:
    def __init__(
        self,
        checkpoint=None,
        randomize=False,
        device=None,
        seed=0,
        record=None,
        clean=False,
        rec_size=(REC_W, REC_H),
        rec_fps=REC_FPS,
        rec_crf=REC_CRF,
        rec_preset=REC_PRESET,
        rec_overlay=True,
        window_size=(WINDOW_W, WINDOW_H),
    ):
        self.env = WheeledLeggedEnv(
            randomize=randomize,
            max_steps=1 << 60,  # never truncate; fall or BACKSPACE ends an episode
            command_resample_seconds=1e12,
            push_prob=0.0,  # P applies shoves explicitly instead
            seed=seed,
        )

        self.drivers = []
        if checkpoint:
            from vect_ppo import load_policy

            fn = load_policy(checkpoint, device=device)
            tag = os.path.splitext(os.path.basename(checkpoint))[0]
            self.drivers.append(
                PolicyDriver(fn, f"{tag} ({fn.global_step / 1e6:.1f}M steps)")
            )
        self.drivers.append(BaselineDriver())
        self.driver_idx = 0

        self.command = np.array([0.0, 0.0, C.HEIGHT_MAX])
        self.held = set()
        self.falls = 0
        self.distance = 0.0
        self.sim_time = 0.0

        # Display toggles
        self.show_hud = self.show_caps = self.show_keys = not clean

        # Recording: rendered offscreen at a fixed size, never read off the
        # window, so DPI / resizing / tiling / minimising do not affect it.
        self.rec_path = record
        self.rec_size = tuple(rec_size)
        self.rec_fps = rec_fps
        self.rec_crf = rec_crf
        self.rec_preset = rec_preset
        self.rec_overlay = rec_overlay  # burn the HUD / caps / keys into the video
        self.window_size = tuple(window_size)
        self.writer = None
        self.frame = self.clean = None  # offscreen pixels, with / without overlay
        self.rec_seq = 0

        # Mouse camera state.
        self._btn_left = self._btn_right = False
        self._last_xy = (0.0, 0.0)

    @property
    def driver(self):
        return self.drivers[self.driver_idx]

    # --- input ---
    def _on_key(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            self.held.add(key)
            if key in (glfw.KEY_ESCAPE, glfw.KEY_Q):
                glfw.set_window_should_close(window, True)
            elif key == glfw.KEY_SPACE:
                self.command[0] = self.command[1] = 0.0
            elif key == glfw.KEY_TAB:
                self.driver_idx = (self.driver_idx + 1) % len(self.drivers)
                self.driver.reset()
                self._clamp_command()
            elif key == glfw.KEY_P:
                self.env.data.qvel[0:2] += self.env.rng.normal(0.0, SHOVE_SIGMA, size=2)
            elif key == glfw.KEY_BACKSPACE:
                self.reset()
            elif key == glfw.KEY_F1:
                self.show_hud = not self.show_hud
            elif key == glfw.KEY_F2:
                self.show_caps = not self.show_caps
            elif key == glfw.KEY_F3:
                self.show_keys = not self.show_keys
            elif key == glfw.KEY_F4:
                self.rec_overlay = not self.rec_overlay
                print(f"recording overlay {'on' if self.rec_overlay else 'off'}")
            elif key == glfw.KEY_F9:
                self._toggle_record()
        elif action == glfw.RELEASE:
            self.held.discard(key)

    def _bounds(self):
        d = self.driver
        return (-d.max_speed, d.max_speed), (-d.max_yaw, d.max_yaw)

    def _clamp_command(self):
        vel, yaw = self._bounds()
        self.command[0] = _clamp(self.command[0], vel)
        self.command[1] = _clamp(self.command[1], yaw)

    def _update_command(self, dt):
        held, c = self.held, self.command
        vel_b, yaw_b = self._bounds()
        fwd = (glfw.KEY_W in held) - (glfw.KEY_S in held)
        turn = (glfw.KEY_A in held) - (glfw.KEY_D in held)
        lift = (glfw.KEY_R in held) - (glfw.KEY_F in held)

        c[0] = (
            _clamp(c[0] + fwd * VEL_RATE * dt, vel_b)
            if fwd
            else _toward_zero(c[0], VEL_DECAY * dt)
        )
        c[1] = (
            _clamp(c[1] + turn * YAW_RATE * dt, yaw_b)
            if turn
            else _toward_zero(c[1], YAW_DECAY * dt)
        )
        if lift:
            c[2] = _clamp(c[2] + lift * HEIGHT_RATE * dt, C.CMD_HEIGHT_RANGE)

    def _on_mouse_button(self, window, button, action, mods):
        self._btn_left = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        )
        self._btn_right = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        )
        self._last_xy = glfw.get_cursor_pos(window)

    def _on_mouse_move(self, window, x, y):
        if not (self._btn_left or self._btn_right):
            self._last_xy = (x, y)
            return
        dx, dy = x - self._last_xy[0], y - self._last_xy[1]
        self._last_xy = (x, y)
        h = max(1, glfw.get_window_size(window)[1])
        act = (
            mj.mjtMouse.mjMOUSE_ROTATE_V
            if self._btn_left
            else mj.mjtMouse.mjMOUSE_MOVE_V
        )
        mj.mjv_moveCamera(self.env.model, act, dx / h, dy / h, self.scn, self.cam)

    def _on_scroll(self, window, xoff, yoff):
        mj.mjv_moveCamera(
            self.env.model,
            mj.mjtMouse.mjMOUSE_ZOOM,
            0.0,
            -0.05 * yoff,
            self.scn,
            self.cam,
        )

    # --- episode ---
    def reset(self):
        self.obs = self.env.reset(command=self.command)
        self.driver.reset()

    def _control_step(self):
        self.env.set_command(self.command)
        action = self.driver.act(self.obs, self.command, self.env.data)
        self.obs, _, terminated, _, _ = self.env.step(action)
        self.distance += abs(C.FORWARD_SIGN * self.env._v_body[0, 1]) * C.CONTROL_DT
        if terminated:
            self.falls += 1
            self.reset()

    # --- recording ---
    def _toggle_record(self):
        if self.writer is not None:
            path, frames = self.writer.close()
            self.writer = None
            print(f"recorded {frames} frames ({frames / self.rec_fps:.1f} s) -> {path}")
            return
        path = self.rec_path or "videos/teleop.mp4"
        if self.rec_seq:
            root, ext = os.path.splitext(path)
            path = f"{root}_{self.rec_seq + 1:02d}{ext}"
        self.rec_seq += 1
        w, h = self.rec_size
        try:
            self.writer = VideoWriter(
                path, w, h, self.rec_fps, self.rec_crf, self.rec_preset
            )
        except RuntimeError as exc:
            print(f"recording unavailable: {exc}")
            return
        overlay = "with overlay" if self.rec_overlay else "clean"
        print(
            f"recording {w}x{h} at {self.rec_fps:g} fps, {overlay} -> {path}  "
            "(F9 to stop, F4 toggles overlay)"
        )

    def _render(self):
        """Render one frame offscreen at the video size and read its pixels.

        This is the only scene render per frame: the same pixels go to the
        video and (scaled to fit) to the window, like recorder.py does.
        """
        w, h = self.rec_size
        rect = mj.MjrRect(0, 0, w, h)
        mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_OFFSCREEN, self.ctx)
        mj.mjr_render(rect, self.scn, self.ctx)
        clean_needed = self.writer is not None and not self.rec_overlay
        if clean_needed:
            mj.mjr_readPixels(self.clean, None, rect, self.ctx)
            mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_OFFSCREEN, self.ctx)
        self._overlay(rect)
        mj.mjr_readPixels(self.frame, None, rect, self.ctx)
        if self.writer is not None:
            self.writer.add(self.clean if clean_needed else self.frame)

    def _present(self, window):
        """Show the rendered frame, fitted to the window without cropping."""
        width, height = glfw.get_framebuffer_size(window)
        if width <= 0 or height <= 0:
            return  # minimised: recording carries on offscreen
        frame = self.frame
        fh, fw = frame.shape[:2]
        scale = min(width / fw, height / fh)
        fit_w, fit_h = max(1, round(fw * scale)), max(1, round(fh * scale))
        dest = mj.MjrRect((width - fit_w) // 2, (height - fit_h) // 2, fit_w, fit_h)
        if (fit_w, fit_h) != (fw, fh):
            frame = np.asarray(
                Image.fromarray(frame).resize((fit_w, fit_h), Image.Resampling.BILINEAR)
            )
        mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_WINDOW, self.ctx)
        mj.mjr_rectangle(mj.MjrRect(0, 0, width, height), 0, 0, 0, 1)
        mj.mjr_drawPixels(np.ascontiguousarray(frame).ravel(), None, dest, self.ctx)

    # -- rendering ---
    def _overlay(self, viewport):
        if self.show_hud:
            c = self.command
            v = C.FORWARD_SIGN * self.env._v_body[0, 1]
            yaw = self.env._w_body[0, 2]
            h = self.env.data.qpos[2]
            d = self.driver
            rec = (
                f"REC {self.writer.frames / self.rec_fps:.1f}s"
                if self.writer
                else "not recording"
            )
            mj.mjr_overlay(
                mj.mjtFont.mjFONT_NORMAL,
                mj.mjtGridPos.mjGRID_TOPLEFT,
                viewport,
                "controller\ntop speed\n\nforward\nturn\nheight\n\nuptime\nfalls\nF9",
                f"[{self.driver_idx + 1}/{len(self.drivers)}] {d.name}\n"
                f"{d.max_speed:.1f} m/s\n\n"
                f"{c[0]:+5.2f} -> {v:+5.2f} m/s\n"
                f"{c[1]:+5.2f} -> {yaw:+5.2f} rad/s\n"
                f"{c[2]:5.2f} -> {h:5.2f} m\n\n"
                f"{self.env.step_count * C.CONTROL_DT:.0f} s\n{self.falls}\n{rec}",
                self.ctx,
            )
        if self.show_keys:
            mj.mjr_overlay(
                mj.mjtFont.mjFONT_NORMAL,
                mj.mjtGridPos.mjGRID_BOTTOMLEFT,
                viewport,
                "W/S\nA/D\nR/F\nSPACE\nTAB\nP\nBACKSPACE\nF1/F2/F3\nF4\nF9\nESC",
                "drive\nturn\nheight\nstop\nswitch controller\nshove\nreset\n"
                "readout / caps / this list\noverlay in recording\nrecord\nquit",
                self.ctx,
            )
        if self.show_caps:
            self._keycaps(viewport)

    def _keycaps(self, viewport):
        """Light-up key caps"""
        size = max(28, min(52, viewport.width // 24, viewport.height // 13))
        gap = max(2, size // 12)
        margin = max(10, size // 3)
        pitch = size + gap
        rows = max(r for r, *_ in KEYCAPS) + 1
        left = viewport.left + viewport.width - 4 * pitch - margin
        top = viewport.bottom + margin + rows * pitch

        for row, col, span, label, codes, one_shot in KEYCAPS:
            pressed = any(code in self.held for code in codes)
            bg = (CAP_SHOT if one_shot else CAP_HELD) if pressed else CAP_IDLE
            rect = mj.MjrRect(
                left + col * pitch,
                top - (row + 1) * pitch,
                span * size + (span - 1) * gap,
                size,
            )
            mj.mjr_label(
                rect, mj.mjtFont.mjFONT_NORMAL, label, *bg, 1.0, 1.0, 1.0, 1.0, self.ctx
            )

    def run(self):
        if not glfw.init():
            raise RuntimeError("glfw.init() failed - is a display available?")
        window = glfw.create_window(
            *self.window_size, "wheeled-legged teleop", None, None
        )
        if not window:
            glfw.terminate()
            raise RuntimeError("could not create a GLFW window")
        glfw.make_context_current(window)
        glfw.swap_interval(0)  # the loop paces frames itself

        model = self.env.model
        # The offscreen buffer is sized when the context is made, so set the
        # video resolution first. It is purely visual; physics is unaffected.
        model.vis.global_.offwidth, model.vis.global_.offheight = self.rec_size
        model.vis.quality.offsamples = REC_SAMPLES
        self.cam = mj.MjvCamera()
        self.opt = mj.MjvOption()
        mj.mjv_defaultCamera(self.cam)
        mj.mjv_defaultOption(self.opt)  # every render flag at its default
        self.scn = mj.MjvScene(model, maxgeom=10_000)
        self.ctx = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150)
        if (self.ctx.offWidth, self.ctx.offHeight) != self.rec_size:
            print(
                f"warning: offscreen buffer is {self.ctx.offWidth}x"
                f"{self.ctx.offHeight}, wanted {self.rec_size[0]}x{self.rec_size[1]}"
            )
        w, h = self.rec_size
        self.frame = np.empty((h, w, 3), dtype=np.uint8)
        self.clean = np.empty((h, w, 3), dtype=np.uint8)

        self.cam.type = mj.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "base")
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 5.0, 90.0, -12.0

        glfw.set_key_callback(window, self._on_key)
        glfw.set_mouse_button_callback(window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(window, self._on_mouse_move)
        glfw.set_scroll_callback(window, self._on_scroll)

        self.reset()
        if self.rec_path:
            self._toggle_record()
        try:
            # Like recorder.py: every loop is exactly one video frame and
            # advances the sim by exactly 1/fps, so the video is smooth and
            # real-time even if rendering / encoding briefly falls behind
            # (the live view slows down instead of the video skipping).
            frame_dt = 1.0 / self.rec_fps
            sim_target = 0.0
            while not glfw.window_should_close(window):
                frame_started = time.perf_counter()
                glfw.poll_events()
                sim_target += frame_dt
                while self.sim_time + 1e-9 < sim_target:
                    self._update_command(C.CONTROL_DT)
                    self._control_step()
                    self.sim_time += C.CONTROL_DT

                mj.mjv_updateScene(
                    model,
                    self.env.data,
                    self.opt,
                    None,
                    self.cam,
                    mj.mjtCatBit.mjCAT_ALL,
                    self.scn,
                )
                self._render()
                self._present(window)
                glfw.swap_buffers(window)
                delay = frame_dt - (time.perf_counter() - frame_started)
                if delay > 0:
                    time.sleep(delay)
        except KeyboardInterrupt:
            pass
        finally:
            if self.writer is not None:
                path, frames = self.writer.close()
                print(
                    f"recorded {frames} frames ({frames / self.rec_fps:.1f} s) -> {path}"
                )
            self.ctx.free()
            glfw.terminate()
        print(
            f"drove {self.distance:.1f} m over {self.sim_time:.0f} s of sim, "
            f"{self.falls} fall(s)"
        )


def main():
    p = argparse.ArgumentParser(
        description="Keyboard teleoperation of the wheeled-legged robot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "checkpoint",
        nargs="?",
        default=None,
        help="PPO checkpoint to drive. Defaults to the newest in runs/; "
        "pass 'none' for the analytic baseline only.",
    )
    p.add_argument(
        "--randomize",
        action="store_true",
        help="randomised dynamics and start state, as in training",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="policy device (default cpu: one robot at 50 Hz does not "
        "need a GPU, and avoids a ~1 s CUDA init)",
    )
    p.add_argument(
        "--record",
        metavar="PATH",
        nargs="?",
        const="videos/teleop.mp4",
        default=None,
        help="record from the start to PATH "
        "(default videos/teleop.mp4); F9 also toggles",
    )
    p.add_argument(
        "--clean",
        action="store_true",
        help="start with every overlay hidden, on screen and in the video",
    )
    p.add_argument(
        "--no-overlay",
        action="store_true",
        help="keep the overlay on screen but record only the robot; F4 toggles",
    )
    p.add_argument(
        "--width",
        type=int,
        default=REC_W,
        help=f"video width, independent of the window (default {REC_W})",
    )
    p.add_argument(
        "--height",
        type=int,
        default=REC_H,
        help=f"video height, independent of the window (default {REC_H})",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=REC_FPS,
        help=f"video and display frame rate (default {REC_FPS}, matches the 50 Hz controller)",
    )
    p.add_argument(
        "--crf",
        type=int,
        default=REC_CRF,
        help=f"x264 quality, 0-51, lower is better (default {REC_CRF})",
    )
    p.add_argument(
        "--preset", default=REC_PRESET, help=f"x264 preset (default {REC_PRESET})"
    )
    p.add_argument("--window-width", type=int, default=WINDOW_W)
    p.add_argument("--window-height", type=int, default=WINDOW_H)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if any(v < 2 or v % 2 for v in (args.width, args.height)):
        p.error(
            "--width and --height must be positive even integers (e.g. 1920 x 1080)"
        )
    if not (np.isfinite(args.fps) and args.fps > 0):
        p.error("--fps must be finite and positive")
    if not 0 <= args.crf <= 51:
        p.error("--crf must be between 0 and 51")
    if min(args.window_width, args.window_height) < 1:
        p.error("window dimensions must be positive")

    ck = args.checkpoint
    if ck is None:
        ck = newest_checkpoint()
        if ck:
            print(f"driving the newest checkpoint: {ck}")
        else:
            print("no checkpoint in runs/ - analytic baseline only")
    elif ck.lower() == "none":
        ck = None

    t = Teleop(
        checkpoint=ck,
        randomize=args.randomize,
        device=args.device,
        seed=args.seed,
        record=args.record,
        clean=args.clean,
        rec_size=(args.width, args.height),
        rec_fps=args.fps,
        rec_crf=args.crf,
        rec_preset=args.preset,
        rec_overlay=not args.no_overlay,
        window_size=(args.window_width, args.window_height),
    )
    print(__doc__.split("\n\n")[1].rstrip())
    print("\ncontrollers (TAB to switch):")
    for i, d in enumerate(t.drivers, 1):
        print(f"  {i}. {d.name:<34} up to {d.max_speed:.1f} m/s")
    t.run()


if __name__ == "__main__":
    main()
