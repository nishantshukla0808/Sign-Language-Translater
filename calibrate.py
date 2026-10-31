"""
--calibrate: the user holds still for a few seconds; we measure the wrist
velocity distribution the camera + tracker produce at rest and put
start_velocity just above it.

Thresholds are per camera AND per frame rate (velocity is shoulder-widths per
FRAME), so the config records the measured fps and realtime.py warns on a
mismatch. Re-run on every new camera / on the Pi.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from gate import MotionGate

DEFAULT_PATH = "calibration.json"
MIN_START = 0.006          # never go below this even for a perfectly steady tracker
START_HEADROOM = 1.25      # x the 99th percentile of resting jitter
MAX_PLAUSIBLE_START = 0.05  # warn above this: shoulder-widths/frame at rest is not this large
STOP_RATIO = 0.4           # stop_velocity = start_velocity * ratio (hysteresis)


def compute_thresholds(velocities: np.ndarray) -> dict:
    v = np.asarray(velocities, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 10:
        raise ValueError(f"only {v.size} usable velocity samples; need >= 10 "
                         "(is the signer in frame with shoulders visible?)")
    p50, p95, p99, vmax = (float(np.percentile(v, q)) for q in (50, 95, 99, 100))
    start = max(MIN_START, p99 * START_HEADROOM)
    return {
        "start_velocity": round(start, 5),
        "stop_velocity": round(start * STOP_RATIO, 5),
        "rest_velocity": {"n": int(v.size), "p50": round(p50, 5), "p95": round(p95, 5),
                          "p99": round(p99, 5), "max": round(vmax, 5)},
    }


def velocities_from_frames(frames) -> np.ndarray:
    """Per-frame wrist velocity, computed by the exact function the gate uses."""
    g = MotionGate(start_velocity=1e9)       # never triggers
    out = []
    for f in frames:
        g.push(f)
        out.append(g.last_velocity)
    return np.asarray(out[1:])               # first frame has no previous


def run_calibration(read_frame, seconds: float = 5.0, countdown: float = 2.0,
                    out_path: str | Path = DEFAULT_PATH, camera_info: dict | None = None) -> dict:
    """
    read_frame() -> (543,3) frame or None at end of stream. Blocks `seconds`.
    """
    print(f"[calibrate] hold STILL with both shoulders in view. Starting in {countdown:.0f}s ...")
    time.sleep(countdown)
    print(f"[calibrate] sampling {seconds:.0f}s -- stay still")
    frames, t0 = [], time.monotonic()
    while time.monotonic() - t0 < seconds:
        f = read_frame()
        if f is None:
            break
        frames.append(f)
    elapsed = time.monotonic() - t0
    fps = len(frames) / elapsed if elapsed > 0 else 0.0
    res = compute_thresholds(velocities_from_frames(frames))
    res.update({"measured_fps": round(fps, 2), "frames": len(frames),
                "seconds": round(elapsed, 2), "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "camera": camera_info or {}})
    if res["start_velocity"] > MAX_PLAUSIBLE_START:
        print(f"[calibrate] WARNING: resting jitter is very high (start_velocity "
              f"{res['start_velocity']} > {MAX_PLAUSIBLE_START}). Either you moved during sampling "
              f"or the tracker is unstable here (lighting? hands/arms in frame?). Re-run before trusting this.")
        res["warning"] = "high resting velocity; re-run"
    Path(out_path).write_text(json.dumps(res, indent=2))
    print(f"[calibrate] {len(frames)} frames @ {fps:.1f} fps; resting jitter "
          f"p99={res['rest_velocity']['p99']} -> start_velocity={res['start_velocity']} "
          f"stop_velocity={res['stop_velocity']}  (written to {out_path})")
    return res


def load_calibration(path: str | Path | None, run_fps: float | None = None) -> dict | None:
    if not path or not Path(path).exists():
        return None
    cfg = json.loads(Path(path).read_text())
    mf = cfg.get("measured_fps")
    if run_fps and mf and abs(run_fps - mf) / mf > 0.25:
        print(f"[calibrate] WARNING: {path} was measured at {mf} fps but this run targets "
              f"{run_fps} fps; velocity thresholds are per-frame, re-run --calibrate.")
    return cfg
