"""
Adapter: our (T, 543, 3) landmark window -> ASL Citizen ST-GCN input (2, 128, 27),
plus an ONNX runtime wrapper with the same interface as classifier.SignClassifier.

No PyTorch at runtime. The .onnx comes from export_stgcn.py.

Preprocessing is a line-for-line mirror of
vendor/ASL-citizen-code/ST-GCN/asl_citizen_dataset_pose.py (ASLCitizen.__getitem__)
and of how its keypoints were extracted (ST-GCN/pose.py), including two quirks
that matter and are deliberately kept (see NORM_MODES):

  1. Landmarks the detector did not find were stored as 0.0 in training
     (pose.py allocates np.zeros and only writes what MediaPipe Holistic found).
     So NaN -> 0.0 here, BEFORE normalisation. This is the opposite of the
     PopSign model, which wants NaN.
  2. The shoulder centre / scale are averaged over the zero-PADDED 128-frame
     array, not over the real frames. For a clip of T<128 frames the centre is
     shrunk by T/128 and the scale inflated by 128/T. That is what the network
     was trained on, so "train" mode reproduces it. "valid" mode computes the
     statistics over real frames only; it is offered as an experiment, it is
     NOT what the checkpoint saw.

Node layout of the 27-node graph (after reorder + keypoint selection):
    0-6   pose  [0, 2, 5, 11, 12, 13, 14]  (nose, eyes, shoulders, elbows)
    7-16  LEFT  hand  [wrist, thumb tip, ...]   <- user's left
    17-26 RIGHT hand                           <- user's right
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from landmarks import LHAND_OFF, POSE_OFF, RHAND_OFF, N_LHAND, N_POSE, N_RHAND

MAX_FRAMES = 128
KEYPOINTS = [0, 2, 5, 11, 12, 13, 14, 33, 37, 38, 41, 42, 45, 46, 49, 50, 53,
             54, 58, 59, 62, 63, 66, 67, 70, 71, 74]
N_CLASSES = 2731
NORM_MODES = ("train", "valid")

# Node index ranges in the final (2, T, 27) tensor.
NODES_POSE = range(0, 7)
NODES_LHAND = range(7, 17)
NODES_RHAND = range(17, 27)


def downsample(frames: np.ndarray, max_frames: int) -> np.ndarray:
    """Verbatim from ST-GCN/asl_citizen_dataset_pose.py."""
    length = frames.shape[0]
    increment = max_frames / length
    if increment > 1.0:
        increment = 1.0
    curr_increment = 0
    curr_frame = 0
    new_frames = []
    for f in frames:
        curr_increment += increment
        if curr_increment > curr_frame:
            curr_frame += 1
            new_frames.append(f)
    if len(new_frames) > max_frames:
        new_frames = new_frames[:max_frames]
    return np.array(new_frames)


def window_to_raw75(window: np.ndarray) -> np.ndarray:
    """
    (T, 543, 3) [Face, LHand, Pose, RHand] -> (T, 75, 2) in the *training* layout:
        pose 0-32, RIGHT hand 33-53, LEFT hand 54-74. x,y only. NaN -> 0.0.
    The hand order differs from ours; test_pipeline.py fails if they are swapped.
    """
    if window.ndim != 3 or window.shape[1:] != (543, 3):
        raise ValueError(f"expected (T, 543, 3), got {window.shape}")
    pose = window[:, POSE_OFF:POSE_OFF + N_POSE, :2]
    rhand = window[:, RHAND_OFF:RHAND_OFF + N_RHAND, :2]
    lhand = window[:, LHAND_OFF:LHAND_OFF + N_LHAND, :2]
    raw = np.concatenate([pose, rhand, lhand], axis=1).astype(np.float64)
    return np.nan_to_num(raw, nan=0.0)


def preprocess(window: np.ndarray, norm: str = "train") -> np.ndarray:
    """(T, 543, 3) -> (2, 128, 27) float32, ready for the ONNX model."""
    if norm not in NORM_MODES:
        raise ValueError(f"norm must be one of {NORM_MODES}")
    data0 = window_to_raw75(window)
    length = data0.shape[0]
    if length == 0:
        raise ValueError("empty window")
    if length > MAX_FRAMES:
        data0 = downsample(data0, MAX_FRAMES)
    n_valid = data0.shape[0]   # real frames (after any downsampling), before padding
    if data0.shape[0] < MAX_FRAMES:
        data0 = np.pad(data0, ((0, MAX_FRAMES - data0.shape[0]), (0, 0), (0, 0)))

    sl, sr = data0[:, 11, :], data0[:, 12, :]
    stat = slice(None) if norm == "train" else slice(0, n_valid)
    center = ((sl[stat] + sr[stat]) / 2).mean(axis=0)
    mean_dist = np.mean(np.sqrt(((sl[stat] - sr[stat]) ** 2).sum(-1)))
    if mean_dist != 0:
        data0 = (data0 - center) * (1.0 / mean_dist)

    pose, rh, lh = data0[:, 0:33], data0[:, 33:54], data0[:, 54:75]
    data = np.concatenate([pose, lh, rh], axis=1)[:, KEYPOINTS, :]
    return np.ascontiguousarray(np.transpose(data, (2, 0, 1)), dtype=np.float32)


def load_gloss_map(path: str | Path | None, n: int = N_CLASSES) -> dict[int, str]:
    """
    {index: name}. The real 2,731 glosses are the alphabetically sorted gloss
    column of the ASL Citizen CSVs, which are not available here, so the default
    is "sign_<index>". Drop a real {"0": "...", ...} JSON at the path to swap it.
    """
    if path and Path(path).exists():
        raw = json.loads(Path(path).read_text())
        if isinstance(raw, list):
            return {i: str(g) for i, g in enumerate(raw)}
        return {int(k): str(v) for k, v in raw.items()}
    return {i: f"sign_{i}" for i in range(n)}


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class AslCitizenClassifier:
    """ONNX ST-GCN wrapper. Same surface as classifier.SignClassifier."""

    name = "aslcitizen"

    def __init__(self, model_path: str | Path, gloss_path: str | Path | None = None,
                 threads: int = 2, norm: str = "train"):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(model_path), opts,
                                         providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.norm = norm
        self.idx_to_sign = load_gloss_map(gloss_path)
        self.vocab_size = int(self.sess.get_outputs()[0].shape[-1] or N_CLASSES)
        self.labels_are_real = bool(gloss_path and Path(gloss_path).exists())
        if not self.labels_are_real:
            print(f"[aslcitizen] NOTE: real gloss names unavailable -> labels are "
                  f"'sign_<index>' (0..{self.vocab_size - 1}). Supply "
                  f"models/aslcitizen_gloss.json ({{index: name}}) to swap them in.")

    def predict(self, window: np.ndarray) -> np.ndarray:
        x = preprocess(window, self.norm)[None]
        logits = self.sess.run(None, {self.input_name: x})[0]
        return softmax(np.asarray(logits, dtype=np.float64).ravel())

    def top_k(self, probs: np.ndarray, k: int = 2) -> list[tuple[str, float]]:
        idx = np.argsort(probs)[::-1][:k]
        return [(self.idx_to_sign.get(int(i), f"sign_{int(i)}"), float(probs[i])) for i in idx]

    def warmup(self, frames: int = 32) -> float:
        import time
        t0 = time.perf_counter()
        dummy = np.full((frames, 543, 3), np.nan, dtype=np.float32)
        self.predict(dummy)
        return time.perf_counter() - t0
