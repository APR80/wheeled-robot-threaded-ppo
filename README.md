# Wheeled-Legged Robot: Fast Parallel PPO

A PPO implementation, built on a threaded vectorised MuJoCo environment, that trains a wheeled-legged robot to balance and follow velocity, yaw and ride-height commands. It is designed for throughput on modest hardware

A hand-tuned analytic controller is included as a baseline to compare the learned policy against.

[![Wheeled-Legged Robot: Fast Parallel PPO](https://img.youtube.com/vi/jld0LdgmoZQ/maxresdefault.jpg)](https://youtu.be/jld0LdgmoZQ)

[https://youtu.be/jld0LdgmoZQ](https://youtu.be/jld0LdgmoZQ)

## What it does

The policy receives a command `(forward velocity, yaw rate, ride height)` and outputs hip position targets and wheel torques, keeping the robot upright while it follows the command. The same command vector is what you drive from the keyboard in `teleop.py`.

## Why it's fast

- **Threaded vectorised env (`vectorized.py`).** MuJoCo releases the GIL, so physics runs in a thread pool. All observation, reward and termination work is done once per vector-step as batch numpy kernels on the main thread, because per-env Python work gets slower under GIL contention.
- **One model per thread, not per env.** Domain randomisation is applied by writing each env's mass and friction into its thread's model just before stepping. Memory per env drops from about 6.5 MB to 0.5 MB, which makes hundreds of envs practical.
- **Wide rather than deep rollouts.** PPO step cost is dominated by fixed GPU latency, so more envs per step means higher throughput (13k steps/s at 48 envs, 28.7k at 256, 33k at 512).
- **Lean PPO loop (`vect_ppo.py`).** Hand-rolled Gaussian, preallocated rollout buffers, no device syncs inside the epoch loop, fused Adam.

## Correctness details

- Time-limit truncation is bootstrapped instead of treated as a terminal state.
- State-independent `log_std`.
- Running-mean/std normalisation for observations and value targets.
- Domain randomisation (mass, friction, start state) plus random pushes.

## Files

| File | Purpose |
|---|---|
| `vect_ppo.py` | PPO trainer and CLI |
| `vectorized.py` | Threaded vectorised environment |
| `wheeled_leged_env.py` | Single env and batch obs/reward/termination kernels |
| `robot_config.py` | Robot and task constants, command ranges, observation scales |
| `baseline_controller.py` | Analytic cascade controller (baseline) |
| `teleop.py` | Keyboard control of a trained policy or the baseline |
| `recorder.py` | Offscreen 1080p video recording with a fixed command schedule |

## Usage

Requirements: Python 3, `mujoco`, `numpy`, `torch`, `glfw`, and `ffmpeg` for recording.

```bash
# Train
python vect_ppo.py --envs 256 --steps 40000000

# Drive a trained policy (TAB switches between policy and baseline)
python teleop.py runs/<checkpoint>.pt

# Record a checkpoint or the baseline
python recorder.py --checkpoint runs/<checkpoint>.pt
python recorder.py                      # baseline
python teleop.py --record --no-overlay  # record without overlay
```

Teleop keys: `W/S` drive, `A/D` turn, `R/F` height, `SPACE` stop, `TAB` switch controller, `P` shove, `BACKSPACE` reset, `F9` record.

## Note
This project was done before mjx came out but you can still use it if you want to use regular mujoco.
