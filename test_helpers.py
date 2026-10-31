"""Synthetic-frame builders shared by test_pipeline.py and eval_clips.py --selftest.
Plumbing only: these say nothing about recognition accuracy."""
from landmarks import POSE_L_SHOULDER, POSE_L_WRIST, POSE_R_SHOULDER, POSE_R_WRIST, assemble_frame


class LM:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


def synth_frame(wrist_x: float, with_hands: bool = True):
    """A frame with shoulders fixed and wrists at a given x."""
    hands = []
    if with_hands:
        hands = [
            type("H", (), {"handedness": "Left",
                           "landmarks": [LM(wrist_x + i * 0.001, 0.5) for i in range(21)],
                           "score": 0.9})(),
            type("H", (), {"handedness": "Right",
                           "landmarks": [LM(wrist_x + 0.2 + i * 0.001, 0.5) for i in range(21)],
                           "score": 0.9})(),
        ]
    pose = [LM(0.5, 0.5) for _ in range(33)]
    pose[POSE_L_SHOULDER] = LM(0.40, 0.40)
    pose[POSE_R_SHOULDER] = LM(0.60, 0.40)   # shoulder width = 0.20
    pose[POSE_L_WRIST] = LM(wrist_x, 0.50)
    pose[POSE_R_WRIST] = LM(wrist_x + 0.2, 0.50)
    return assemble_frame(face=None, hands=hands, pose=pose)
