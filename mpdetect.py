"""
Shared MediaPipe -> (543, 3) frame path. realtime.py and eval_clips.py both go
through detect_frame(), so live and offline numbers come from one implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from landmarks import HandResult, assemble_frame


@dataclass
class Detectors:
    hands: object
    pose: object
    face: object | None = None

    def close(self) -> None:
        # Explicit close avoids noisy __del__ tracebacks at interpreter shutdown.
        for d in (self.hands, self.pose, self.face):
            if d is not None:
                try:
                    d.close()
                except Exception:
                    pass


def build_detectors(task_dir: Path, use_face: bool) -> Detectors:
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    task_dir = Path(task_dir)

    def opts(cls, fname, **kw):
        base = mp_python.BaseOptions(model_asset_path=str(task_dir / fname))
        return cls(base_options=base, running_mode=vision.RunningMode.VIDEO, **kw)

    hands = vision.HandLandmarker.create_from_options(
        opts(vision.HandLandmarkerOptions, "hand_landmarker.task", num_hands=2))
    pose = vision.PoseLandmarker.create_from_options(
        opts(vision.PoseLandmarkerOptions, "pose_landmarker_lite.task"))
    face = None
    if use_face:
        face = vision.FaceLandmarker.create_from_options(
            opts(vision.FaceLandmarkerOptions, "face_landmarker.task"))
    return Detectors(hands, pose, face)


def to_mp_image(bgr: np.ndarray):
    import cv2
    import mediapipe as mp
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def detect_frame(mp_image, ts_ms: int, det: Detectors) -> np.ndarray:
    """One image -> (543, 3) float32 frame, NaN where not detected."""
    hr = det.hands.detect_for_video(mp_image, ts_ms)
    pr = det.pose.detect_for_video(mp_image, ts_ms)
    fr = det.face.detect_for_video(mp_image, ts_ms) if det.face else None

    hands: list[HandResult] = []
    for lms, hd in zip(hr.hand_landmarks, hr.handedness):
        cat = hd[0]
        hands.append(HandResult(handedness=cat.category_name,
                                landmarks=lms, score=cat.score))

    pose = pr.pose_landmarks[0] if pr.pose_landmarks else None
    face = fr.face_landmarks[0] if (fr and fr.face_landmarks) else None
    return assemble_frame(face=face, hands=hands, pose=pose)


class MonotonicTs:
    """MediaPipe VIDEO mode needs strictly increasing integer ms timestamps."""

    def __init__(self):
        self._last = -1

    def __call__(self, ms: float) -> int:
        t = int(ms)
        if t <= self._last:
            t = self._last + 1
        self._last = t
        return t
