"""
Session logging: one directory per run, one JSONL line + one .npy per emitted
segment. Core feature -- this is how you tell tracking failures from model
failures after the fact (replay the .npy through any model).

    sessions/20260101_120000/
        meta.json            run configuration
        index.jsonl          one line per segment
        seg_0001.npy         (T, 543, 3) float32 raw landmark window (NaN = missing)
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np


class SessionLogger:
    def __init__(self, root: str | Path = "sessions", meta: dict | None = None):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = Path(root) / stamp
        n = 1
        while self.dir.exists():          # two runs inside one second
            n += 1
            self.dir = Path(root) / f"{stamp}_{n}"
        self.dir.mkdir(parents=True)
        (self.dir / "meta.json").write_text(json.dumps(meta or {}, indent=2, default=str))
        self._index = open(self.dir / "index.jsonl", "a", buffering=1)
        self.count = 0

    def log_segment(self, window: np.ndarray, top: list[tuple[str, float]],
                    velocity_trace: np.ndarray, emitted: str | None,
                    via: str | None = None, extra: dict | None = None) -> Path:
        self.count += 1
        name = f"seg_{self.count:04d}.npy"
        np.save(self.dir / name, window.astype(np.float32))
        rec = {
            "seq": self.count,
            "ts_unix": time.time(),
            "ts_iso": datetime.now().isoformat(timespec="milliseconds"),
            "window_file": name,
            "frames": int(window.shape[0]),
            "top5": [{"label": l, "score": round(float(s), 5)} for l, s in top[:5]],
            "emitted": emitted,
            "admitted_via": via,
            "velocity_trace": [round(float(v), 5) for v in np.asarray(velocity_trace)],
        }
        if extra:
            rec.update(extra)
        self._index.write(json.dumps(rec) + "\n")
        return self.dir / name

    def close(self) -> None:
        self._index.close()
