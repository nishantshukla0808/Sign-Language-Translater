"""
Inference-stage logic with no I/O: frame in -> (maybe) classified segment out.
realtime.py's inference thread, eval_clips.py's live-path replay and the tests
all drive this one class.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from classifier import Token, TokenAdmitter
from gate import MotionGate

TOP_N = 5          # logged; the overlay / console show the first 3
DEFAULT_REJECT = 0.10


@dataclass
class Segment:
    window: np.ndarray                 # (T, 543, 3)
    top: list[tuple[str, float]]       # top-5 (label, prob)
    token: Token | None                # committed word, or None (held / rejected)
    velocity_trace: np.ndarray
    infer_ms: float
    rejected: bool                     # top-1 below the confidence floor
    ts: float = field(default_factory=time.time)


class Engine:
    def __init__(self, model, gate: MotionGate, admitter: TokenAdmitter,
                 reject_threshold: float = DEFAULT_REJECT):
        self.model = model
        self.gate = gate
        self.admitter = admitter
        self.reject_threshold = reject_threshold

    def classify(self, window: np.ndarray, velocity_trace: np.ndarray | None = None) -> Segment:
        t0 = time.perf_counter()
        probs = self.model.predict(window)
        top = self.model.top_k(probs, TOP_N)
        infer_ms = (time.perf_counter() - t0) * 1000
        rejected = top[0][1] < self.reject_threshold
        tok = self.admitter.offer_segment(top, reject_threshold=self.reject_threshold)
        vt = velocity_trace if velocity_trace is not None else np.zeros(0, np.float32)
        return Segment(window, top, tok, vt, infer_ms, rejected)

    def push(self, frame: np.ndarray) -> Segment | None:
        window = self.gate.push(frame)
        if window is None:
            return None
        return self.classify(window, self.gate.last_window_velocities)
