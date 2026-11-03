"""
Phrase-network inference shared by the web server and the live test.

    model = PhraseModel()                  # models/phrases.onnx + models/phrases_labels.json
    probs, labels = model.predict(window)  # window: (T,543,3) landmark window, NaN = missing

The input is built exactly as in train_phrases.build_input: shoulder-normalised positions,
velocities x4 and two hand-present flags per frame, 32 frames. The model files are re-read
automatically when they change on disk (after a retrain), keeping the old model if the new
one is unusable.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from fewshot import POSE_IDX, T_LEN, features
from landmarks import LHAND_OFF, N_TOTAL, POSE_OFF, RHAND_OFF

ROOT = Path(__file__).resolve().parent
TRAIN_ASPECT = 16 / 9            # training clips were 640x360


def build_input(win: np.ndarray) -> np.ndarray:
    base = features(win).reshape(T_LEN, 98).astype(np.float32)
    idx = np.round(np.linspace(0, len(win) - 1, T_LEN)).astype(int)
    flags = np.stack([np.isfinite(win[idx, LHAND_OFF, 0]), np.isfinite(win[idx, RHAND_OFF, 0])], 1).astype(np.float32)
    vel = np.concatenate([np.zeros((1, 98), np.float32), base[1:] - base[:-1]], 0) * 4.0
    return np.concatenate([base, vel, flags], -1)[None]


def window_from_payload(frames, aspect: float) -> np.ndarray:
    """Browser landmarks -> (T,543,3) window.

    Each frame is [pose, left_hand, right_hand]: pose = 7 (x,y) pairs for POSE_IDX (14 numbers),
    hands = 21 (x,y) pairs (42 numbers), any of them null when not detected. x is rescaled so a
    phone image of any aspect ratio looks like the 16:9 webcam images the model was trained on.
    """
    if not isinstance(frames, list) or not 1 <= len(frames) <= 600:
        raise ValueError("frames must be a list of 1..600 frames")
    aspect = float(aspect)
    if not 0.3 <= aspect <= 4.0:
        raise ValueError("bad aspect ratio")
    kx = aspect / TRAIN_ASPECT
    win = np.full((len(frames), N_TOTAL, 3), np.nan, np.float32)
    pose_rows = [POSE_OFF + i for i in POSE_IDX]
    try:
        for t, fr in enumerate(frames):
            if len(fr) != 3:
                raise ValueError("frame must have 3 parts")
            for rows, vals, n in ((pose_rows, fr[0], 7), (slice(LHAND_OFF, LHAND_OFF + 21), fr[1], 21),
                                  (slice(RHAND_OFF, RHAND_OFF + 21), fr[2], 21)):
                if vals is None:
                    continue
                a = np.asarray(vals, np.float32)
                if a.shape != (2 * n,):
                    raise ValueError("bad landmark count")
                a = a.reshape(n, 2)
                win[t, rows, 0] = a[:, 0] * kx
                win[t, rows, 1] = a[:, 1]
                win[t, rows, 2] = 0.0
    except (TypeError, KeyError) as e:
        raise ValueError(f"malformed frames: {e}")
    return win


def quality(win: np.ndarray) -> tuple[float, float]:
    """(fraction of frames with at least one hand, fraction with both shoulders)."""
    hand = np.isfinite(win[:, LHAND_OFF, 0]) | np.isfinite(win[:, RHAND_OFF, 0])
    body = np.isfinite(win[:, POSE_OFF + 11, 0]) & np.isfinite(win[:, POSE_OFF + 12, 0])
    return float(hand.mean()), float(body.mean())


class PhraseModel:
    def __init__(self, model_dir: Path | str = ROOT / "models", threads: int = 2):
        self.dir = Path(model_dir)
        self.threads = threads
        self.error = ""
        self._lock = threading.Lock()
        self._checked = 0.0
        self._m = self._load()
        self._stamp = self._mtimes()
        self.loaded_at = time.time()

    def _mtimes(self):
        return (self.dir / "phrases.onnx").stat().st_mtime_ns, (self.dir / "phrases_labels.json").stat().st_mtime_ns

    def _load(self):
        so = ort.SessionOptions()
        so.intra_op_num_threads = self.threads
        so.inter_op_num_threads = 1
        sess = ort.InferenceSession(str(self.dir / "phrases.onnx"), so, providers=["CPUExecutionProvider"])
        labels = json.loads((self.dir / "phrases_labels.json").read_text())
        out = sess.run(None, {"x": np.zeros((1, T_LEN, 198), np.float32)})[0]      # also warms the session
        if out.shape[1] != len(labels):
            raise ValueError(f"model has {out.shape[1]} outputs but {len(labels)} labels")
        return sess, labels

    @property
    def labels(self) -> list[str]:
        return self._m[1]

    def maybe_reload(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._checked < 1.0:
            return
        self._checked = now
        try:
            stamp = self._mtimes()
        except OSError:
            return
        if stamp == self._stamp:
            return
        with self._lock:
            if stamp == self._stamp:
                return
            self._stamp = stamp                     # do not retry a bad file until it changes again
            try:
                self._m = self._load()
                self.loaded_at = time.time()
                self.error = ""
                print(f"[model] reloaded: {len(self._m[1])} phrases", flush=True)
            except Exception as e:                  # keep serving the old model
                self.error = str(e)
                print(f"[model] reload failed, keeping the old model: {e}", flush=True)

    def predict(self, win: np.ndarray) -> tuple[np.ndarray, list[str]]:
        self.maybe_reload()
        sess, labels = self._m
        z = sess.run(None, {"x": build_input(win)})[0][0].astype(np.float64)
        e = np.exp(z - z.max())
        return e / e.sum(), labels
