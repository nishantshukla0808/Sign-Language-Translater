"""
MediaPipe Tasks -> (T, 543, 3) landmark tensor for the asl-signs ONNX model.

CRITICAL SPEC (reverse-engineered from the model + signchat's TS implementation):

  Row order:  [Face(468)] [LHand(21)] [Pose(33)] [RHand(21)]  = 543 rows
  Per row:    (x, y, z) float32

  Missing landmarks are NaN, NOT 0.
  The model has normalization BAKED IN -- it normalizes finite values per clip
  and treats NaN as "not detected". Writing zeros would silently move
  "not detected" into the data distribution and wreck accuracy.

  x, y  : normalized image coords in [0, 1]
  z     : depth in image-width units, smaller = closer to camera
  Face  : Tasks Vision FaceLandmarker returns 478 (468 face + 10 iris).
          Truncate to the first 468 -- those match the layout the model saw.
  Hands : handedness is relative to the USER, not the camera. MediaPipe and
          the training data agree, so "Left" -> LHand block, no flip needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

N_FACE, N_LHAND, N_POSE, N_RHAND = 468, 21, 33, 21
N_TOTAL = N_FACE + N_LHAND + N_POSE + N_RHAND  # 543

FACE_OFF = 0
LHAND_OFF = N_FACE                       # 468
POSE_OFF = N_FACE + N_LHAND              # 489
RHAND_OFF = N_FACE + N_LHAND + N_POSE    # 522

# Pose landmark indices (MediaPipe pose, 33 points) used by the motion gate.
POSE_L_WRIST, POSE_R_WRIST = 15, 16
POSE_L_SHOULDER, POSE_R_SHOULDER = 11, 12


@dataclass
class HandResult:
    handedness: str                  # "Left" | "Right" (relative to the user)
    landmarks: Sequence             # 21 objects with .x .y .z
    score: float = 1.0


def new_frame() -> np.ndarray:
    """A fresh (543, 3) frame filled with NaN."""
    return np.full((N_TOTAL, 3), np.nan, dtype=np.float32)


def _write(frame: np.ndarray, offset: int, count: int, landmarks) -> None:
    if not landmarks:
        return
    limit = min(count, len(landmarks))
    for i in range(limit):
        lm = landmarks[i]
        frame[offset + i, 0] = lm.x
        frame[offset + i, 1] = lm.y
        frame[offset + i, 2] = getattr(lm, "z", 0.0)


def _pick_hand(hands: Iterable[HandResult], side: str) -> HandResult | None:
    """Highest-scoring hand for a given side; MediaPipe occasionally returns two."""
    best = None
    for h in hands:
        if h.handedness != side:
            continue
        if best is None or h.score > best.score:
            best = h
    return best


def assemble_frame(face=None, hands: Sequence[HandResult] = (), pose=None) -> np.ndarray:
    """Build one (543, 3) frame. Anything not detected stays NaN."""
    frame = new_frame()
    if face:
        _write(frame, FACE_OFF, N_FACE, face)          # truncates 478 -> 468
    lh = _pick_hand(hands, "Left")
    rh = _pick_hand(hands, "Right")
    if lh:
        _write(frame, LHAND_OFF, N_LHAND, lh.landmarks)
    if rh:
        _write(frame, RHAND_OFF, N_RHAND, rh.landmarks)
    if pose:
        _write(frame, POSE_OFF, N_POSE, pose)
    return frame


def detected_points(frame: np.ndarray) -> int:
    """How many of the 543 rows are present. Diagnostics / sanity checks."""
    return int(np.isfinite(frame[:, 0]).sum())


def hands_present(frame: np.ndarray) -> int:
    """0, 1 or 2. Used by the motion gate to know when signing can start."""
    n = 0
    if np.isfinite(frame[LHAND_OFF, 0]):
        n += 1
    if np.isfinite(frame[RHAND_OFF, 0]):
        n += 1
    return n


class FrameRing:
    """Bounded chronological buffer of frames (oldest first)."""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._items: list[np.ndarray] = []

    def push(self, frame: np.ndarray) -> None:
        if frame.shape != (N_TOTAL, 3):
            raise ValueError(f"frame shape {frame.shape}, expected ({N_TOTAL}, 3)")
        self._items.append(frame)
        if len(self._items) > self.capacity:
            self._items.pop(0)

    def stack(self) -> np.ndarray:
        """(T, 543, 3) float32 ready for the ONNX session."""
        if not self._items:
            return np.empty((0, N_TOTAL, 3), dtype=np.float32)
        return np.stack(self._items).astype(np.float32, copy=False)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)
