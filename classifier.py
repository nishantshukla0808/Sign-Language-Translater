"""
ONNX classifier wrapper + token admission.

The model outputs RAW LOGITS (250,), not probabilities -- softmax before
comparing against thresholds.

Admission logic ported from signchat's admit.ts. Single-tick top-1 predictions
are noisy; a label is only committed when either:
  "stable" - score >= top1_threshold AND same label as the previous tick
  "band"   - score in [top2_threshold, top1_threshold) with a credible
             runner-up, and not a repeat of the last admitted label
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort

# Defaults from signchat's contracts/mode.ts. Tune against live use, not a test set.
TOP1_THRESHOLD = 0.30
TOP2_THRESHOLD = 0.10
INFERENCE_INTERVAL_MS = 200
SILENCE_MS = 500


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


@dataclass
class Token:
    label: str
    score: float
    via: str          # "stable" | "band"
    ts: float = field(default_factory=time.monotonic)


class SignClassifier:
    """PopSign-250 ONNX model. Takes (T, 543, 3) windows directly."""

    name = "popsign"
    labels_are_real = True

    def __init__(self, model_path: str | Path, vocab_path: str | Path, threads: int = 2):
        opts = ort.SessionOptions()
        # Pi 5 has 4 cores; leave headroom for MediaPipe, which is the real cost.
        opts.intra_op_num_threads = threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(
            str(model_path), opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.sess.get_inputs()[0].name

        sign_to_idx = json.loads(Path(vocab_path).read_text())
        self.idx_to_sign = {v: k for k, v in sign_to_idx.items()}
        self.vocab_size = len(sign_to_idx)

    def predict(self, window: np.ndarray) -> np.ndarray:
        """(T, 543, 3) -> (250,) probabilities."""
        if window.ndim != 3 or window.shape[1:] != (543, 3):
            raise ValueError(f"expected (T, 543, 3), got {window.shape}")
        logits = self.sess.run(None, {self.input_name: window.astype(np.float32)})[0]
        return softmax(np.asarray(logits).ravel())

    def top_k(self, probs: np.ndarray, k: int = 2) -> list[tuple[str, float]]:
        idx = np.argsort(probs)[::-1][:k]
        return [(self.idx_to_sign[int(i)], float(probs[i])) for i in idx]

    def warmup(self, frames: int = 32) -> float:
        """
        One dummy inference so graph optimisation / allocator warm-up is paid
        at startup rather than on the first real sign. Returns seconds taken.
        """
        t0 = time.perf_counter()
        dummy = np.full((frames, 543, 3), np.nan, dtype=np.float32)
        dummy[:, 468:489, :] = 0.5
        self.predict(dummy)
        return time.perf_counter() - t0


class TokenAdmitter:
    """Turns a stream of noisy per-tick predictions into committed words."""

    def __init__(
        self,
        top1_threshold: float = TOP1_THRESHOLD,
        top2_threshold: float = TOP2_THRESHOLD,
        refractory_s: float = 1.0,
    ):
        self.top1_threshold = top1_threshold
        self.top2_threshold = top2_threshold
        self.refractory_s = refractory_s
        self.tokens: list[Token] = []
        self._prev_top_label: str | None = None
        self._last_admit_at = 0.0

    def offer(self, candidates: list[tuple[str, float]],
              reject_threshold: float = 0.0) -> Token | None:
        """Feed one tick's top-2. Returns a Token if a word was committed."""
        (l1, s1), (l2, s2) = candidates[0], candidates[1]
        if s1 < reject_threshold:
            self._prev_top_label = l1
            return None
        last = self.tokens[-1] if self.tokens else None
        now = time.monotonic()

        via = None
        if s1 >= self.top1_threshold and l1 == self._prev_top_label:
            via = "stable"
        elif (
            self.top2_threshold <= s1 < self.top1_threshold
            and s2 >= self.top2_threshold
            and (last is None or l1 != last.label)
        ):
            via = "band"

        self._prev_top_label = l1

        if via is None:
            return None
        # Refractory period stops one long sign firing the same word repeatedly.
        if now - self._last_admit_at < self.refractory_s:
            return None

        tok = Token(label=l1, score=s1, via=via, ts=now)
        self.tokens.append(tok)
        self._last_admit_at = now
        return tok

    def offer_segment(
        self, candidates: list[tuple[str, float]], reject_threshold: float = 0.0,
    ) -> Token | None:
        """
        Admission for ONE prediction per gate-segmented sign.

        offer() was written for a stream of ticks: "stable" needs the same label
        on two consecutive ticks. With one prediction per segment that would mean
        a confident word only commits when the signer repeats it, so a confident
        segment is committed directly here (via="confident"). Band logic is
        unchanged. Below `reject_threshold` nothing is ever emitted -- the model
        always returns one of N classes, this is the "none of the above" floor.
        """
        (l1, s1), s2 = candidates[0], (candidates[1][1] if len(candidates) > 1 else 0.0)
        self._prev_top_label = l1
        if s1 < reject_threshold:
            return None
        last = self.tokens[-1] if self.tokens else None
        via = None
        if s1 >= self.top1_threshold:
            via = "confident"
        elif (self.top2_threshold <= s1 < self.top1_threshold
              and s2 >= self.top2_threshold
              and (last is None or l1 != last.label)):
            via = "band"
        if via is None:
            return None
        now = time.monotonic()
        if now - self._last_admit_at < self.refractory_s:
            return None
        tok = Token(label=l1, score=s1, via=via, ts=now)
        self.tokens.append(tok)
        self._last_admit_at = now
        return tok

    def reset(self) -> None:
        self.tokens.clear()
        self._prev_top_label = None
        self._last_admit_at = 0.0
