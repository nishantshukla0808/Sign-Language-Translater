"""
Motion gate: decides when a sign starts and ends.

Why this exists: the model was trained on pre-trimmed clips. Live video is a
continuous stream. Without gating you classify half-signs and the output
machine-guns garbage words. This is the single biggest difference between a
demo that looks broken and one that looks real.

Approach: track wrist velocity in shoulder-normalized units (so it is invariant
to how far the signer sits from the camera), then run a small state machine.
"""

from __future__ import annotations

import numpy as np

from landmarks import (
    POSE_L_SHOULDER,
    POSE_L_WRIST,
    POSE_OFF,
    POSE_R_SHOULDER,
    POSE_R_WRIST,
    hands_present,
)

IDLE, RECORDING = "idle", "recording"


def _shoulder_width(frame: np.ndarray) -> float | None:
    ls = frame[POSE_OFF + POSE_L_SHOULDER, :2]
    rs = frame[POSE_OFF + POSE_R_SHOULDER, :2]
    if not (np.isfinite(ls).all() and np.isfinite(rs).all()):
        return None
    w = float(np.linalg.norm(ls - rs))
    return w if w > 1e-4 else None


def _wrists(frame: np.ndarray) -> np.ndarray:
    """(2, 2) wrist xy; NaN rows where not detected."""
    return np.stack([
        frame[POSE_OFF + POSE_L_WRIST, :2],
        frame[POSE_OFF + POSE_R_WRIST, :2],
    ])


class MotionGate:
    """
    Emits a window of frames each time a sign completes.

    start_velocity / stop_velocity are in shoulder-widths per frame. Defaults
    are a starting point -- tune them on your own camera and frame rate, they
    are the knobs that matter most.
    """

    def __init__(
        self,
        start_velocity: float = 0.020,
        stop_velocity: float = 0.008,
        start_frames: int = 3,
        stop_frames: int = 8,
        min_frames: int = 10,
        max_frames: int = 90,
    ):
        self.start_velocity = start_velocity
        self.stop_velocity = stop_velocity
        self.start_frames = start_frames
        self.stop_frames = stop_frames
        self.min_frames = min_frames
        self.max_frames = max_frames

        self.state = IDLE
        self._prev_wrists: np.ndarray | None = None
        self._fast_run = 0
        self._slow_run = 0
        self._buffer: list[np.ndarray] = []
        self._vbuf: list[float] = []
        self.last_velocity = 0.0
        self.last_window_velocities: np.ndarray = np.zeros(0, dtype=np.float32)

    def _velocity(self, frame: np.ndarray) -> float:
        w = _wrists(frame)
        sw = _shoulder_width(frame)
        prev = self._prev_wrists
        self._prev_wrists = w
        if prev is None or sw is None:
            return 0.0
        d = np.linalg.norm(w - prev, axis=1) / sw     # per-wrist, normalized
        d = d[np.isfinite(d)]
        return float(d.max()) if d.size else 0.0

    def push(self, frame: np.ndarray) -> np.ndarray | None:
        """
        Feed one frame. Returns a (T, 543, 3) window when a sign just ended,
        otherwise None.
        """
        v = self._velocity(frame)
        self.last_velocity = v
        have_hands = hands_present(frame) > 0

        if self.state == IDLE:
            self._fast_run = self._fast_run + 1 if (v >= self.start_velocity and have_hands) else 0
            if self._fast_run >= self.start_frames:
                self.state = RECORDING
                self._buffer = [frame]
                self._vbuf = [v]
                self._slow_run = 0
                self._fast_run = 0
            return None

        # RECORDING
        self._buffer.append(frame)
        self._vbuf.append(v)
        self._slow_run = self._slow_run + 1 if v < self.stop_velocity else 0

        ended = self._slow_run >= self.stop_frames
        overlong = len(self._buffer) >= self.max_frames
        if not (ended or overlong):
            return None

        frames, vs = self._buffer, self._vbuf
        self._buffer, self._vbuf = [], []
        self.state = IDLE
        self._slow_run = 0

        if ended:
            # Trim the trailing stillness that triggered the stop.
            keep = max(self.min_frames, len(frames) - self.stop_frames)
            frames, vs = frames[:keep], vs[:keep]
        return self._emit(frames, vs)

    def _emit(self, frames, vs):
        if len(frames) < self.min_frames:
            return None
        self.last_window_velocities = np.asarray(vs, dtype=np.float32)
        return np.stack(frames).astype(np.float32)

    def flush(self) -> np.ndarray | None:
        """
        End of a recorded clip: emit whatever is being recorded. The live loop
        never needs this (it waits for stillness); offline clips often end the
        instant the sign does.
        """
        if self.state != RECORDING:
            return None
        frames, vs = self._buffer, self._vbuf
        self._buffer, self._vbuf = [], []
        self.state = IDLE
        self._slow_run = 0
        return self._emit(frames, vs)

    def reset(self) -> None:
        self.state = IDLE
        self._buffer = []
        self._vbuf = []
        self._prev_wrists = None
        self._fast_run = self._slow_run = 0
