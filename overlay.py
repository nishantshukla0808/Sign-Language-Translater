"""Debug overlay for `realtime.py --show`. Imported only when --show is set."""

from __future__ import annotations

import cv2
import numpy as np

from landmarks import LHAND_OFF, POSE_OFF, RHAND_OFF

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
              (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
# upper body only; legs are never relevant to signing
POSE_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]

C_LEFT, C_RIGHT, C_POSE = (255, 180, 0), (0, 200, 255), (80, 255, 80)   # BGR


def _px(pt, w, h):
    return int(pt[0] * w), int(pt[1] * h)


def _draw_set(img, pts, edges, color, radius):
    h, w = img.shape[:2]
    ok = np.isfinite(pts[:, 0])
    for a, b in edges:
        if ok[a] and ok[b]:
            cv2.line(img, _px(pts[a], w, h), _px(pts[b], w, h), color, 2, cv2.LINE_AA)
    for i in np.flatnonzero(ok):
        cv2.circle(img, _px(pts[i], w, h), radius, color, -1, cv2.LINE_AA)


def draw_overlay(img: np.ndarray, frame543: np.ndarray, gate_state: str, velocity: float,
                 start_velocity: float, top3: list[tuple[str, float]] | None,
                 fps: float, extra: str = "") -> np.ndarray:
    """Draw in place and return img."""
    _draw_set(img, frame543[POSE_OFF:POSE_OFF + 33], POSE_EDGES, C_POSE, 3)
    _draw_set(img, frame543[LHAND_OFF:LHAND_OFF + 21], HAND_EDGES, C_LEFT, 3)
    _draw_set(img, frame543[RHAND_OFF:RHAND_OFF + 21], HAND_EDGES, C_RIGHT, 3)

    f = cv2.FONT_HERSHEY_SIMPLEX
    col = (0, 0, 255) if gate_state == "recording" else (0, 255, 0)
    cv2.putText(img, f"{gate_state}  v={velocity:.3f} (start>={start_velocity:.3f})",
                (10, 24), f, 0.6, col, 2)
    cv2.putText(img, f"{fps:4.1f} fps  L={'Y' if np.isfinite(frame543[LHAND_OFF, 0]) else '-'}"
                     f" R={'Y' if np.isfinite(frame543[RHAND_OFF, 0]) else '-'} {extra}",
                (10, 48), f, 0.55, (255, 255, 255), 1)
    y = img.shape[0] - 14 - 22 * 2
    cv2.putText(img, "last segment:" if top3 else "last segment: (none yet)", (10, y - 22),
                f, 0.5, (200, 200, 200), 1)
    for i, (lab, sc) in enumerate(top3 or []):
        cv2.putText(img, f"{i + 1}. {lab[:22]:<22} {sc:5.2f}", (10, y + 22 * i),
                    f, 0.55, (255, 255, 255) if i else (0, 255, 255), 1 if i else 2)
    return img
